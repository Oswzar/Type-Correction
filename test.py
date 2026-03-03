from src.model.bert_corrector import DistilBertTypoCorrector
from src.utils.logger import setup_logger


def run_test() -> None:
    """Simple test case to verify typo correction end-to-end."""
    logger = setup_logger()
    logger.info("Starting correction test")

    corrector = DistilBertTypoCorrector(logger=logger)
    sample_input = "helo worl"
    expected_output = "hello world"

    corrected = corrector.correct_text(sample_input)

    logger.info("Test input: %r", sample_input)
    logger.info("Model output: %r", corrected)
    logger.info("Expected output: %r", expected_output)

    if corrected == expected_output:
        logger.info("TEST PASSED")
    else:
        logger.error("TEST FAILED")
        raise AssertionError(f"Expected {expected_output!r}, got {corrected!r}")


if __name__ == "__main__":
    run_test()
