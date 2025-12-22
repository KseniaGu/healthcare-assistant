import asyncio
import json
import os.path
from datetime import timezone
from functools import partial

import fitz
from dotenv import load_dotenv, find_dotenv
from gliner import GLiNER
from langfuse import observe, get_client
from tqdm import tqdm
from transformers import AutoTokenizer

from backend.models.llm_clients.gemini import GeminiClient
from backend.models.llm_clients.llama_parse import LlamaParseClient
from backend.utils.common_funcs import write_file, read_file, get_logger
from backend.utils.exceptions import *
from backend.utils.helpers import generate_artifact_meta, get_hash_and_size
from backend.utils.prompt_engine import PromptEngine
from backend.utils.schemas import *
from configs.constants import GLINER_MAX_WINDOW_SIZE, GLINER_TOKENS_OVERLAP, DEFAULT_LAB_TEST_NAME, \
    DATA_PROCESSING_LOGGING
from configs.data import DataSettings
from configs.database import PostgreSQLSettings
from configs.enums import PromptTypeEnum, TaskEnum, StepTypeEnum, TestFlagTypeEnum
from configs.models import ModelSettings
from configs.paths import PathSettings
from src.databases.postgres.database import PostgreSQLDB

local_logger, memory_handler = get_logger(DATA_PROCESSING_LOGGING)
load_dotenv(find_dotenv())


class LaboratoryTestProcessor:
    """A class for processing laboratory analysis results."""

    def __init__(
            self,
            model_settings: ModelSettings,
            data_settings: DataSettings,
            path_settings: PathSettings,
            database_settings: PostgreSQLSettings
    ):
        self.model_settings = model_settings
        self.data_settings = data_settings
        self.path_settings = path_settings
        self.database = PostgreSQLDB(database_settings)

        # PII extraction
        self.pii_extractor = GLiNER.from_pretrained(self.model_settings.pii_extractor.model_path)
        self.pii_extractor_tokenizer = AutoTokenizer.from_pretrained(self.model_settings.pii_extractor.tokenizer_path)

        # Anonymized documents parsing
        self.document_parser = LlamaParseClient(self.model_settings.document_parser)

        # Parsed data processing
        self.prompt_engine = PromptEngine(self.path_settings.prompts_dir)
        self.data_processor = GeminiClient(self.model_settings.data_processor, self.path_settings)

        # Logging
        self.logger = get_client()

    @observe(name="LaboratoryTestProcessor.cap_text_tokens_count", capture_input=False, capture_output=False)
    def cap_text_tokens_count(self, text: str, token_count: int = GLINER_MAX_WINDOW_SIZE):
        """Chunking solution suggested by https://github.com/urchade/GLiNER/discussions/113#discussioncomment-9956716."""
        token_scale_down_ratio = 5.0
        if len(text) > token_count * token_scale_down_ratio:
            text = text[:int(token_count * token_scale_down_ratio)]

        try:
            tokenized = self.pii_extractor_tokenizer(text, add_special_tokens=False)
            detokenized = self.pii_extractor_tokenizer.decode(
                tokenized['input_ids'][:token_count], skip_special_tokens=True
            )
        except Exception as e:
            raise PIIMaskingError(f"An error occurred while splitting the input text into chunks: {str(e)}")

        return text[:len(detokenized)]

    def get_text_chunks(self, text: str):
        """Splits text into chunks if text is longer than GLINER_MAX_WINDOW_SIZE."""
        cursor_index, chunks = 0, []

        for i in range(0, len(text) - GLINER_TOKENS_OVERLAP, GLINER_MAX_WINDOW_SIZE - GLINER_TOKENS_OVERLAP):
            window_text = self.cap_text_tokens_count(text[cursor_index:])
            chunks.append(window_text)

            if len(window_text) <= GLINER_TOKENS_OVERLAP:
                print('breaking ...')
                break
            cursor_index += len(window_text) - GLINER_TOKENS_OVERLAP

        return chunks

    @staticmethod
    def mask_images(images: list, page: fitz.Page):
        """Masks out (with black rectangular) images found in the given document page."""
        for image in tqdm(images, desc='Masking images...'):
            if image.get('bbox') is not None:
                page.add_redact_annot(quad=image['bbox'], text="IMAGE", fill=(0, 0, 0))

    def mask_entities(self, chunks: list[dict], page: fitz.Page) -> bool:
        """Detects and masks out (with black rectangular) PII entities.

        Returns boolean indicating whether PII was found or not.
        """
        entities_found = False

        for chunk in tqdm(chunks, desc='Masking PII entities...'):
            try:
                entities = self.pii_extractor.predict_entities(
                    chunk, self.data_settings.pii_entities, threshold=self.data_settings.pii_prediction_threshold
                )
                if entities:
                    entities_found = True
            except Exception as e:
                message = f"Exception while predicting entities: {str(e)}"
                local_logger.error(message)
                self.logger.update_current_span(level="ERROR", status_message=message)
                if len(chunk) <= GLINER_TOKENS_OVERLAP:
                    break
                continue

            for entity in tqdm(entities, desc='Processing found entities...'):
                sensitive_text, label = entity["text"], entity["label"]
                areas = page.search_for(sensitive_text)

                for area in areas:
                    page.add_redact_annot(quad=area, text=f"<{label.upper()}>", fill=(0, 0, 0))

        return entities_found

    @observe(name="LaboratoryTestProcessor.mask_pii")
    def mask_pii(self, input_path: Path, output_path: Path) -> PipelineStepSchema | None:
        """Masks PII data in the PDF document from the <input_path>.

        The method searches for and masks out the following data:
            1. Entities from <self.pii_entities> using NER model.
            2. Images (as they are usually QR-codes that link to results or laboratory seals)
        """
        if os.path.exists(output_path):
            message = "Input file is already anonymized. Skipping..."
            local_logger.info(message)
            self.logger.update_current_span(status_message=message)
            return None

        try:
            document = fitz.open(input_path)
            pii_found = False

            for page in document:
                text = page.get_text()
                chunks = self.get_text_chunks(text)
                images = page.get_image_info()

                self.mask_images(images, page)
                pii_found |= self.mask_entities(chunks, page)

                page.apply_redactions()

            if not pii_found:
                message = "No PII found."
                local_logger.warning(message)
                self.logger.update_current_span(level="WARNING", status_message=message)

            # TODO: Only for development, need temporary path saving in production environment
            document.save(output_path)
        except Exception as e:
            raise PIIMaskingError(f"Failed to anonymize file by path: {input_path}. Error: {str(e)}")

        # Step data for database record
        step_data = PipelineStepSchema(
            step_type=StepTypeEnum.anonymize,
            tool_name=self.model_settings.pii_extractor.tool_name,
            model_version=self.model_settings.pii_extractor.model_path,
            provider=self.model_settings.pii_extractor.provider,
            config={"tokenizer_path": self.model_settings.pii_extractor.tokenizer_path},
            finished_at=datetime.now(timezone.utc),
        )
        log_info = {key: getattr(step_data, key) for key in ("tool_name", "model_version", "provider")}
        self.logger.update_current_span(metadata=log_info)

        return step_data

    @observe(name="LaboratoryTestProcessor.parse_document")
    def parse_document(self, input_path: Path, output_path: Path) -> PipelineStepSchema | None:
        """Parses the document and converts it into a format that can be used by LLMs for further processing."""
        if os.path.exists(output_path):
            message = "Input file is already parsed. Skipping..."
            local_logger.info(message)
            self.logger.update_current_span(status_message=message)
            return None

        try:
            documents = self.document_parser.parse(input_path)
            text_content = "\n".join([doc.text for doc in documents])
            if not text_content.strip("\n "):
                raise DocumentParsingError(f"Parsed empty document. Input path: {input_path}")

            write_file(text_content, output_path)
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse document by path: {input_path}. Error: {str(e)}")

        # Step data for database record
        config = {
            "parse_mode": self.model_settings.document_parser.parse_mode,
            "disable_ocr": self.model_settings.document_parser.disable_ocr,
            "disable_image_extraction": self.model_settings.document_parser.disable_image_extraction,
        }
        step_data = PipelineStepSchema(
            step_type=StepTypeEnum.parse,
            tool_name=self.model_settings.document_parser.tool_name,
            provider=self.model_settings.document_parser.provider,
            config=config,
            finished_at=datetime.now(timezone.utc),
        )

        return step_data

    @observe(name="LaboratoryTestProcessor.process_data", capture_output=False)
    def process_data(self, input_path: Path, output_path: Path, model_name: str) -> (PipelineStepSchema | None, Any):
        """Processes parsed laboratory test results data to get structured output."""
        if os.path.exists(output_path):
            message = "Input file is already processed. Skipping..."
            local_logger.info(message)
            self.logger.update_current_span(status_message=message)
            return None, read_file(output_path)

        try:
            input_text = read_file(input_path)
            if input_text is None:
                raise LaboratoryTestProcessingError(f"File not found. Path: {input_path}")
        except Exception as e:
            raise LaboratoryTestProcessingError(f"Failed to read file by path: {input_path}. Error: {str(e)}")

        prompt_type = PromptTypeEnum.laboratory_test_processor
        prompt_version = self.data_settings.prompt_version
        try:
            # TODO: Add choosing prompt w.r.t. the model selected by the user
            prompt, system_prompt, model_settings = self.prompt_engine.render(
                prompt_type, prompt_version, input_text=input_text
            )
            assert model_settings["model"] == self.data_processor.config.model_name
        except AssertionError:
            raise LaboratoryTestProcessingError(f"""
                The model specified in the prompt settings ({model_settings["model"]}) does not match the one specified 
                in configuration file ({self.data_processor.config.model_name})   .
            """)
        except Exception as e:
            raise LaboratoryTestProcessingError(f"Failed to render prompt (version {prompt_version}). Error: {str(e)}")

        logger_args = {"as_type": "generation", "name": "data_processor.generate"}
        with self.logger.start_as_current_observation(**logger_args) as generation_observation:
            generation_observation.update(model=self.data_processor.config.model_name, model_parameters=model_settings)

            try:
                # Get model generation from the parsed document with laboratory test results
                model_output, usage_metadata, raw_output_path = self.data_processor.generate(
                    prompt, prompt_type, prompt_version, **system_prompt, model_name=model_name
                )
                structured_output = json.loads(model_output.replace("```", "").replace("json", ""))
                write_file(structured_output, output_path)
            except Exception as e:
                raise LaboratoryTestProcessingError(
                    f"Failed to get structured output from the document ({input_path}). Error: {str(e)}"
                )
            generation_observation.update(usage_details=usage_metadata, metadata={"raw_output_path": raw_output_path})

        # Step data for database record
        step_data = PipelineStepSchema(
            step_type=StepTypeEnum.process,
            tool_name=self.model_settings.data_processor.tool_name,
            model_version=self.model_settings.data_processor.model_name,
            prompt_path=f"{prompt_type.value}/{prompt_version}.yaml",
            provider=self.model_settings.data_processor.provider,
            finished_at=datetime.now(timezone.utc),
            extra_data={"llm_raw_output_path": str(raw_output_path), "usage_metadata": usage_metadata},
        )
        log_info = {key: getattr(step_data, key) for key in ("tool_name", "model_version", "provider")}
        log_info |= {"prompt_version": prompt_version}
        self.logger.update_current_span(metadata=log_info)

        return step_data, structured_output

    @observe(name="LaboratoryTestProcessor.log_pipeline_step", capture_input=False, capture_output=False)
    async def log_pipeline_step(
            self,
            pipeline_run_id: int,
            step_data: PipelineStepSchema | None,
            storage_type: str,
            input_artifact_id: int,
            output_path: Path,
            step_type: StepTypeEnum,
    ) -> int:
        """Logs processing pipeline step metadata."""
        self.logger.update_current_span(metadata={"step_type": step_type})
        output_artifact_meta = ArtifactSchema(storage_type=storage_type, storage_path=str(output_path))

        if step_data is not None:
            output_artifact_id = await self.database.add_artifact(output_artifact_meta)
            step_data.input_artifact_id = input_artifact_id
            step_data.output_artifact_id = output_artifact_id
            try:
                await self.database.add_pipeline_step(pipeline_run_id, step_data)
            except Exception as e:
                message = f"Error encountered while adding pipeline step info to database: {str(e)}"
                local_logger.error(message)
                self.logger.update_current_span(level="ERROR", status_message=message)
        else:
            output_artifact_id = await self.database.get_artifact(output_artifact_meta)

        return output_artifact_id

    @observe(name="LaboratoryTestProcessor.save_report", capture_input=False, capture_output=False)
    async def save_report(
            self,
            pipeline_run_id: int,
            patient_id: int,
            input_path: Path,
            structured_output_path: Path,
            structured_output: dict | list,
            processed_at: datetime,
            source_artifact_id: int
    ):
        """Saves report (structured output obtained by the processor), extracts and saves test observations."""
        if isinstance(structured_output, dict):
            report_results = {"results": structured_output.get("results", [])}
            report_date = structured_output.get("date", None)
        elif isinstance(structured_output, list):
            report_results = {"results": structured_output}
            report_date = None
        else:
            raise LaboratoryTestProcessingError(f"Processed report data has unexpected type: {type(structured_output)}")

        try:
            report_hash, _ = get_hash_and_size(structured_output_path)
            report_data = ReportCreateSchema(
                report_date=report_date,
                source_path=input_path.name,
                report_hash=report_hash,
                processed_at=processed_at,
                raw_json=report_results,
                source_artifact_id=source_artifact_id,
            )
        except Exception as e:
            raise LaboratoryTestProcessingError(f"Error encountered while getting report hash: {str(e)}")

        report = await self.database.add_report(pipeline_run_id, patient_id, report_data)

        # Extract and save test observations
        test_observations = []
        for i, result in enumerate(report_results["results"]):
            try:
                test_name = result.get("name", "")
                if not test_name:
                    test_name = f"{DEFAULT_LAB_TEST_NAME}_{report.uuid}_{i}"
                    message = f"Test with unknown name. Report uuid: {report.uuid}"
                    local_logger.warning(message)
                    self.logger.update_current_span(level="WARNING", status_message=message)

                test_catalog_data = TestCatalogSchema(
                    canonical_name=test_name,
                    preferred_unit=result.get("units_normalized")
                )
                test_catalog_id = await self.database.get_test_catalog_id(test_catalog_data)
                test_observation = TestObservationSchema(
                    test_catalog_id=test_catalog_id,
                    report_id=report.id,
                    patient_id=patient_id,
                    test_name=test_name,
                    observed_value=result.get("value"),
                    unit=result.get("units"),
                    flag=TestFlagTypeEnum[result.get("flag", "").lower()].name,
                    reference_range=result.get("reference_range"),
                    observation_date=report_date,
                    created_at=processed_at,
                )
                test_observations.append(test_observation)
            except Exception as e:
                message = f"Exception while parsing test observation: {str(e)}"
                local_logger.error(message)
                self.logger.update_current_span(level="ERROR", status_message=message)

        await self.database.add_test_observations(test_observations)

    async def anonymize(self, input_path: Path, patient_id: int, storage_type: str = "local") -> PipelineRunContext:
        """Makes document anonymization process."""
        file_extension = input_path.suffix
        anonymized_output_file = input_path.name.replace(file_extension, f"_anonymized{file_extension}")
        anonymized_output_path = self.path_settings.anonymized_documents_dir / anonymized_output_file

        # Add source file artifact and check for uniqueness
        artifact_metadata = generate_artifact_meta(
            input_path, storage_type=storage_type, exception_to_raise=DocumentParsingError
        )
        input_artifact_id = await self.database.add_artifact(artifact_metadata)
        if input_artifact_id is None:
            # Artifact was not added as it's already in the database
            input_artifact_id = await self.database.get_artifact(ArtifactSchema(checksum=artifact_metadata.checksum))
            structured_output, pipeline_run_id = await self.database.get_report(
                ReportFilterSchema(source_artifact_id=input_artifact_id)
            )
            if structured_output:
                context = PipelineRunContext(
                    pipeline_run_id=pipeline_run_id,
                    patient_id=patient_id,
                    source_path=input_path,
                    source_artifact_id=input_artifact_id,
                    anonymized_path=anonymized_output_path,
                    report_data=structured_output
                )
                return context

        # Add started pipeline run to the database
        pipeline_run_id, pipeline_run_uuid = await self.database.add_pipeline_run(
            patient_id, task=TaskEnum.laboratory_results
        )
        loop = asyncio.get_running_loop()

        observation_args = {"as_type": "span", "name": "LaboratoryTestProcessor.anonymize"}
        with self.logger.start_as_current_observation(**observation_args) as observation:
            observation.update(metadata={"pipeline_run_id_uuid": pipeline_run_uuid})

            # Mask out PII
            step_data = await loop.run_in_executor(
                None, partial(self.mask_pii, input_path, anonymized_output_path)
            )
            output_artifact_id = await self.log_pipeline_step(
                pipeline_run_id, step_data, storage_type, input_artifact_id, anonymized_output_path,
                StepTypeEnum.anonymize
            )

        context = PipelineRunContext(
            pipeline_run_id=pipeline_run_id,
            pipeline_run_uuid=pipeline_run_uuid,
            patient_id=patient_id,
            source_path=input_path,
            source_artifact_id=input_artifact_id,
            anonymized_path=anonymized_output_path,
            anonymized_artifact_id=output_artifact_id,
        )

        return context

    async def run(self, pipeline_context: PipelineRunContext, storage_type: str = "local") -> dict:
        """Processes laboratory analysis results."""
        loop = asyncio.get_running_loop()
        pipeline_run_id = pipeline_context.pipeline_run_id
        output_artifact_id = pipeline_context.anonymized_artifact_id
        input_path = pipeline_context.source_path
        file_extension = input_path.suffix
        processor_model_name = pipeline_context.processor_model_name

        observation_args = {"as_type": "span", "name": "LaboratoryTestProcessor.run"}
        with self.logger.start_as_current_observation(**observation_args) as observation:
            observation.update(metadata={"pipeline_run_id_uuid": pipeline_context.pipeline_run_uuid})

            # Parse anonymized documents to extract the medical data in a proper text format
            parsed_output_file = input_path.name.replace(file_extension, "_parsed.txt")
            parsed_output_path = self.path_settings.parsed_documents_dir / parsed_output_file
            step_data = await loop.run_in_executor(
                None, partial(self.parse_document, pipeline_context.anonymized_path, parsed_output_path)
            )
            output_artifact_id = await self.log_pipeline_step(
                pipeline_run_id, step_data, storage_type, output_artifact_id, parsed_output_path, StepTypeEnum.parse
            )

            # Process parsed text using LLM to get structured output
            processed_output_file = input_path.name.replace(file_extension, "_processed.json")
            processed_output_path = self.path_settings.processed_documents_dir / processed_output_file
            step_data, structured_output = await loop.run_in_executor(
                None, partial(self.process_data, parsed_output_path, processed_output_path, processor_model_name)
            )
            if not structured_output:
                raise LaboratoryTestProcessingError("The document's structured output is empty")

            await self.log_pipeline_step(
                pipeline_run_id, step_data, storage_type, output_artifact_id, processed_output_path,
                StepTypeEnum.process
            )

            # Finish pipeline and save processed report to database
            finished_at = datetime.now(timezone.utc)
            await self.save_report(
                pipeline_run_id, pipeline_context.patient_id, input_path, processed_output_path, structured_output,
                finished_at, pipeline_context.source_artifact_id
            )
            try:
                await self.database.finish_pipeline_run(pipeline_run_id, finished_at)
            except CriticalDatabaseSideError as e:
                message = f"Failed to finish pipeline run. Exception: {str(e)}"
                local_logger.error(message)
                self.logger.update_current_span(level="ERROR", status_message=message)

        return structured_output


async def main():
    model_config = ModelSettings()
    data_config = DataSettings()
    path_settings = PathSettings()
    database_settings = PostgreSQLSettings()
    LTP = LaboratoryTestProcessor(model_config, data_config, path_settings, database_settings)
    patient_id = await LTP.database.get_patient("base_patient__seed__90ba650e5a6c")
    context = await LTP.anonymize(path_settings.raw_documents_dir / "837453519.pdf", patient_id)
    await LTP.run(context)
    LTP.logger.shutdown()


if __name__ == "__main__":
    ...
    # asyncio.run(main())
