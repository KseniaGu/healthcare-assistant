class CriticalDatabaseSideError(Exception):
    """Base class for errors related to database operations that should be raised and handled properly."""
    user_message = "Database connection issue. We couldn't save your results at this time."
    is_retryable = True


class BaseLLMError(Exception):
    """Base class for errors encountered while generating content using LLMs."""
    user_message = "The AI analysis engine is currently busy. Please try again in a moment."
    is_retryable = True


class DocumentParsingError(Exception):
    """Base class for errors encountered while parsing medical documents."""
    user_message = "We were unable to read the PDF. Please check if the file is valid and not password-protected."
    is_retryable = False


class LaboratoryTestProcessingError(Exception):
    """Base class for all errors from laboratory test processing."""
    user_message = "An unexpected error occurred while processing your document."
    is_retryable = False


class PIIMaskingError(LaboratoryTestProcessingError):
    """Raised when PII extractor fails."""
    user_message = "Security Check Failed: We could not guarantee that all personal data was masked. " \
                   "Processing halted for your protection."
    is_retryable = False
