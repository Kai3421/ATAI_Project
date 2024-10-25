import re
import pyparsing
from speakeasypy import Speakeasy

class ChatBot:
    def __init__(self, knowledge_graph):
        self.speakeasy = None
        self.rooms = []
        self.knowledge_graph = knowledge_graph

    def login(self, host, username, password):
        self.speakeasy = Speakeasy(host=host, username=username, password=password)
        self.speakeasy.login()

    def run(self):
        print("Running the bot...")
        while True:
            self.rooms = self.speakeasy.get_rooms()
            for room in self.rooms:
                if not room.initiated:
                    self.send_message(f"Hi! Let's chat about movies.", room)
                    room.initiated = True
                for message in room.get_messages(only_partner=True, only_new=True):
                    self.process_message(message, room)
                for reaction in room.get_reactions(only_new=True):
                    self.process_reaction(reaction, room)

    def process_message(self, message_object, room):
        if self.is_sparql_query(message_object.message):
            try:
                query_result = self.knowledge_graph.custom_sparql_query(message_object.message)
                self.send_message(f"I found the following query result: \n{query_result}.", room)
            except pyparsing.exceptions.ParseException:
                self.send_message("It looks like you're writing SPARQL, but that seems to be an invalid query.", room)
        else:
            self.send_message(f"For sure.", room)
        room.mark_as_processed(message_object)

    def process_reaction(self, reaction, room):
        self.send_message(f"Received your reaction: '{reaction.type}' ", room)
        room.mark_as_processed(reaction)

    def send_message(self, message, room):
        room.post_messages(self.sanitize(message))

    @staticmethod
    def is_sparql_query(message):
        sparql_keywords = ["SELECT", "ASK", "WHERE", "PREFIX", "DESCRIBE", "CONSTRUCT"]
        has_keyword = any(keyword in message.upper() for keyword in sparql_keywords)
        has_sparql_symbols = bool(re.search(r"\?|<.*?>|\{.*?\}", message))
        return has_keyword and has_sparql_symbols

    @staticmethod
    def sanitize(message):
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
