import logging
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import sys
from datetime import datetime
from chatbot.chatbot import ChatBot
from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


class StreamToLogger:
    """Redirects writes to a logger instead of stdout or stderr. Meaning, when a module tries to print something, it is
    instead saved in a log file."""
    def __init__(self, logger_object, log_level):
        self.logger = logger_object
        self.log_level = log_level
        self.line_buffer = ""

    def write(self, message):
        if message.strip():  # Ignore empty messages
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass  # Required for file-like objects but not needed here


def setup_logging():
    """Sets up logging. Each log file is named after the date and time when the chatbot was started. The log files are
    stored in the folder './logs'."""
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


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("main")
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    logger.info("Starting...")

    knowledge_graph = KnowledgeGraph()
    named_entity_recognizer = NamedEntityRecognizer()
    relation_extractor = RelationExtractor()

    chatbot = ChatBot(knowledge_graph, named_entity_recognizer, relation_extractor)
    host = "https://speakeasy.ifi.uzh.ch"
    username = "playful-panther"
    with open("chatbot/password.txt") as file:
        password = file.readline()
    chatbot.login(host=host, username=username, password=password)

    while True:
        try:
            chatbot.run()
        except Exception as error:
            if isinstance(error, KeyboardInterrupt):
                raise
            else:
                logger.error("An unexpected error occurred.", exc_info=True)
                break
        except KeyboardInterrupt:
            logger.info("Stopped by keyboard interrupt.")
            break
    logger.info("Exiting...")
    chatbot.logout()
