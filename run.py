from src.listener.hotkey_listener import MacTextSelectionListener
from src.model.bert_corrector import DistilBertTypoCorrector, get_default_model_info
from src.utils.logger import setup_logger


def main() -> None:
    """Project entry point: boot model + global listener."""
    logger = setup_logger()

    model_name, model_link = get_default_model_info()
    logger.info("Using model: %s", model_name)
    logger.info("Model link: %s", model_link)

    try:
        corrector = DistilBertTypoCorrector(logger=logger, model_name=model_name)
    except Exception as error:
        logger.error("Model initialization failed: %s", error)
        logger.error("Please check your network and Python package installation")
        return

    listener = MacTextSelectionListener(logger=logger, shortcut="<alt>+<shift>")

    def handle_shortcut() -> None:
        """Capture selected text, run correction, then paste corrected text back."""
        selected_text = listener.get_selected_text()
        if not selected_text:
            logger.warning("No selectable text was captured")
            return

        corrected_text = corrector.correct_text(selected_text)
        if corrected_text == selected_text:
            logger.info("No correction needed")
            return

        success = listener.replace_selected_text(corrected_text)
        if success:
            logger.info("Text replacement completed successfully")
        else:
            logger.error("Text replacement failed")

    listener.set_callback(handle_shortcut)

    logger.info("Type Correction is running. Press Alt+Shift after selecting text.")
    logger.info("Press Ctrl+C in this terminal to stop the program.")
    listener.start()


if __name__ == "__main__":
    main()
    
