import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.utils.common_funcs import write_temp_file
from backend.data_processing.laboratory_test import LaboratoryTestProcessor, local_logger, memory_handler
from configs.constants import BASE_PATIENT_NAME
from configs.data import DataSettings
from configs.database import PostgreSQLSettings
from configs.models import ModelSettings
from configs.paths import PathSettings
from backend.utils.exceptions import *

# Page Configuration
st.set_page_config(
    page_title="Lab Results Analytics",
    page_icon='https://cdn-icons-png.flaticon.com/128/8730/8730564.png',
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stSidebar"] { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True
)


@st.cache_resource
def get_processor():
    """Initializes and caches the heavy backend processor and base patient."""
    processor = LaboratoryTestProcessor(
        ModelSettings(), DataSettings(), PathSettings(), PostgreSQLSettings()
    )
    patient_id = asyncio.run(processor.database.get_patient(BASE_PATIENT_NAME))
    return processor, patient_id


processor, patient_id = get_processor()

if "pipeline_context" not in st.session_state:
    st.session_state.pipeline_context = None

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/128/17626/17626723.png", width=80)
    st.title("Healthcare Assistant")
    st.subheader("Patient Diagnostics")

    page = st.radio("Navigation", ["Process Report", "Lab Test Dynamics", "System Logs"])

    st.divider()
    st.info("**Current Patient:** You! :)")

# Main content area
# Page navigation: Process Report
if page == "Process Report":
    st.header("📄 Lab Report Intake")
    st.caption("Upload a PDF lab report to anonymize and extract clinical data.")

    # Model Selection
    with st.container():
        selected_model = st.selectbox(
            "Select Processing Model",
            options=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3-flash", "gemini-3-pro"],
            index=0,
            help="Higher models provide better extraction accuracy but may be slower."
        )

    # Upload Section
    with st.container(border=True):
        uploaded_file = st.file_uploader("Drop laboratory PDF here", type=["pdf"])

        if uploaded_file and st.session_state.pipeline_context is None:
            if st.button("Start Anonymization"):
                try:
                    with st.status("Anonymizing...", expanded=True) as status:
                        input_path = write_temp_file(uploaded_file.getvalue())

                        # Start anonymizing the user document with laboratory test results
                        context = asyncio.run(processor.anonymize(input_path, patient_id=patient_id))
                        if context.report_data is not None:
                            status.update(label="Report has already been generated!", state="complete")
                            st.json(context.report_data)
                            st.session_state.pipeline_context = None
                        else:
                            context.processor_model_name = selected_model

                            st.session_state.pipeline_context = context
                            status.update(label="Anonymization Complete!", state="complete")
                            st.rerun()
                except (DocumentParsingError, PIIMaskingError, CriticalDatabaseSideError, BaseLLMError) as e:
                    st.error(f"Error: {e.user_message}")
                    if e.is_retryable:
                        if st.button("🔄 Retry Anonymization"):
                            st.rerun()
                except Exception as e:
                    st.error("A critical system error occurred. Please contact support.")
                    message = f"Unexpected error: {str(e)}"
                    local_logger.error(message)
                    processor.logger.update_current_span(level="ERROR", status_message=message)

    # Review Phase (Only shows if context exists)
    if st.session_state.pipeline_context:
        st.divider()
        st.subheader("Verification Required")

        col_pdf, col_actions = st.columns([2, 1])

        with col_pdf:
            st.info("Preview of Anonymized Document")
            st.pdf(st.session_state.pipeline_context.anonymized_path, height=500)

        with col_actions:
            st.warning("Review the mask integrity before saving to EHR.")
            if st.button("✅ Confirm & Analyze", type="primary"):
                try:
                    with st.spinner("Document processing in progress..."):
                        # Start processing anonymized document to get structured output with laboratory test results
                        result = asyncio.run(processor.run(st.session_state.pipeline_context))
                        st.success("Data written to Database!")
                        st.json(result)

                        context = st.session_state.pipeline_context
                        if context is not None:
                            os.remove(context.source_path)
                            st.session_state.pipeline_context = None
                except (BaseLLMError, CriticalDatabaseSideError) as e:
                    st.error(e.user_message)
                    if e.is_retryable:
                        st.button("🔄 Retry Extraction", on_click=st.rerun)
                except LaboratoryTestProcessingError as e:
                    st.error(e.user_message)
                except Exception as e:
                    st.error("Analysis failed unexpectedly.")
                    message = f"Unexpected error: {str(e)}"
                    local_logger.error(message)
                    processor.logger.update_current_span(level="ERROR", status_message=message)

            if st.button("❌ Reject / Delete"):
                context = st.session_state.pipeline_context
                if context is not None:
                    os.remove(context.source_path)
                    os.remove(context.anonymized_path)
                    st.session_state.pipeline_context = None
                st.rerun()

elif page == "Lab Test Dynamics":
    st.header("📊 Patient Health Dynamics")

    # Get all user laboratory test results and aggregate them by unique tests to show the dynamics
    observations = asyncio.run(processor.database.get_patient_observations(patient_id=patient_id))

    if not observations:
        st.warning("No data found. Please process a report first.")
    else:
        data = []
        for obs in observations:
            data.append({
                "Test": obs.test_name,
                "Value": obs.observed_value,
                "Date": (obs.observation_date or obs.created_at).date(),
                "Flag": obs.flag.value if obs.flag else "NORMAL"
            })
        dataframe = pd.DataFrame(data)

        col_filter, col_chart = st.columns([1, 3])

        with col_filter:
            st.write("### Filter Analysis")

            date_range = st.date_input(
                "Select Date Range",
                value=(dataframe["Date"].min(), dataframe["Date"].max()),
                key="dynamics_page_date_range"
            )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = date_range
                available_tests = dataframe[dataframe["Date"].between(start, end)]["Test"].unique()
            else:
                available_tests = []

            selected_tests = st.multiselect(
                "Select Tests",
                options=sorted(available_tests),
                key="dynamics_page_test_select"
            )

            # Final filtered dataset
            if len(available_tests) > 0 and selected_tests:
                mask = (dataframe["Date"].between(start, end)) & (dataframe["Test"].isin(selected_tests))
                f_df = dataframe[mask].sort_values("Date")

                with col_chart:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Data Points", len(f_df))
                    m2.metric("Critical Flags", len(f_df[f_df["Flag"] != "NORMAL"]))
                    m3.metric("Latest Visit", f_df["Date"].max().strftime('%Y-%m-%d'))

                    fig = px.line(
                        f_df, x="Date", y="Value", color="Test",
                        markers=True, template="plotly_white",
                        title="Biomarker Progression"
                    )
                    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with col_chart:
                    st.info("No laboratory tests match the selected date criteria.")

        with st.expander("🔍 Clinical Data Grid"):
            st.dataframe(dataframe.sort_values("Date", ascending=False), use_container_width=True)

# TODO: Only for admin account
elif page == "System Logs":
    st.header("🛠 Pipeline Monitor")
    logs = memory_handler.get_logs()
    if logs:
        st.code(logs, language="log")
    else:
        st.info("No logs generated in this session yet.")

    if st.button("Clear Cache"):
        memory_handler.flush()
        st.success("Logs cache cleared.")
        st.rerun()
