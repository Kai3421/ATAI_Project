import logging
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import sys
from datetime import datetime
from chatbot.chatbot import ChatBot
from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


class Runner:
    def __init__(self):
        self.logger = self.setup_logger()
        self.logger.info("Starting...")

        self.knowledge_graph = KnowledgeGraph()
        self.named_entity_recognizer = NamedEntityRecognizer()
        self.relation_extractor = RelationExtractor()
        self.chatbot = ChatBot(self.knowledge_graph, self.named_entity_recognizer, self.relation_extractor)

    def main(self):
        running = True
        while running:
            try:
                self.chatbot.run()
            except Exception as error:
                if isinstance(error, KeyboardInterrupt):
                    raise
                else:
                    running = False
                    self.logger.error("An unexpected error occurred.", exc_info=True)
                    user_input = input("Stop or restart? ")
                    if user_input.lower() == "restart":
                        self.restart()
                    else:
                        self.stop()
            except KeyboardInterrupt:
                running = False
                self.logger.info("Stopped by keyboard interrupt.")
                user_input = input("Stop or restart? ")
                if user_input.lower() == "restart":
                    self.restart()
                else:
                    self.stop()

    def restart(self):
        self.logger.info("Restarting...")
        self.chatbot.logout()
        self.chatbot.login()
        self.chatbot.clear()
        self.main()

    def stop(self):
        self.logger.info("Exiting...")
        self.chatbot.logout()

    def setup_logger(self):
        """Sets up logging. Each log file is named after the date and time when the chatbot was started. The log files are
        stored in the folder './logs'. A logger object is created and returned. Print statements are captured and streamed
        to the logger."""
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = os.path.join(log_directory, f"chatbot_{timestamp}.log")
        logging.basicConfig(
            level=logging.DEBUG,  # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        logging.captureWarnings(True)

        logger = logging.getLogger("main")

        return logger


if __name__ == "__main__":
    runner = Runner()
    runner.main()
