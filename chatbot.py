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
                    room.post_messages(f"Hi! Let's chat about movies.")
                    room.initiated = True
                for message in room.get_messages(only_partner=True, only_new=True):
                    self.process_message(message, room)
                for reaction in room.get_reactions(only_new=True):
                    self.process_reaction(reaction, room)

    def process_message(self, message, room):
        try:
            query_result = self.knowledge_graph.custom_sparql_query(message.message)
            room.post_messages(str(query_result))
        except pyparsing.exceptions.ParseException:
            room.post_messages("Sorry, that seems to be an invalid SPARQL query.")
        room.mark_as_processed(message)

    def process_reaction(self, reaction, room):
        room.post_messages(f"Received your reaction: '{reaction.type}' ")
        room.mark_as_processed(reaction)