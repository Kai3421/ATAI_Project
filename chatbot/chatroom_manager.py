import logging
from speakeasypy import Speakeasy


class ChatroomManager(Speakeasy):
    def __init__(self, chatbot):
        self.logger = logging.getLogger("chatroom_manager")

        self.host = "https://speakeasy.ifi.uzh.ch"
        self.username = "playful-panther"
        with open("chatbot/password.txt") as file:
            self.password = file.readline()
        super().__init__(host=self.host, username=self.username, password=self.password)
        self.login()
        self.clear()

        self.chatbot = chatbot

        self.rooms = []

    def run(self):
        """Iterate over all the rooms and handle messages and reactions."""
        self.rooms = self.get_rooms()
        for room in self.rooms:
            if not room.initiated:
                self.send_message(f"Hi! Let's chat about movies.", room)
                room.initiated = True

            for message in room.get_messages(only_partner=True, only_new=True):
                self.process_message(message, room)

            for reaction in room.get_reactions(only_new=True):
                self.process_reaction(reaction, room)

    def process_message(self, message_object, room):
        self.logger.debug(f"Processing message in room {room.my_alias}:")
        message_string = message_object.message
        self.logger.debug(f"Message: {message_string}")
        response = self.chatbot.respond_to(message_string)
        self.logger.debug(f"Response: {response}")
        self.send_message(response, room)
        room.mark_as_processed(message_object)

    def process_reaction(self, reaction, room):
        self.send_message(f"Received your reaction: '{reaction.type}' ", room)
        room.mark_as_processed(reaction)

    def send_message(self, message, room):
        """Sanitizes the message and then sends it to the room."""
        room.post_messages(self.sanitize(message))

    @staticmethod
    def sanitize(message):
        """Replaces problematic characters in the message and encodes it in latin-1"""
        replacements = {
            '–': '-',
            '—': '-',
            '“': '"',
            '”': '"',
            '‘': "'",
            '’': "'",
            '…': '...',
        }
        for original, replacement in replacements.items():
            message = message.replace(original, replacement)

        return message.encode("latin-1", errors="replace").decode("latin-1")

    def clear(self):
        self.rooms = self.get_rooms()
        for room in self.rooms:
            for message in room.get_messages(only_partner=True, only_new=True):
                room.mark_as_processed(message)

            for reaction in room.get_reactions(only_new=True):
                room.mark_as_processed(reaction)
