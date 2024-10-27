import re
import pyparsing
from rdflib.compare import similar
from speakeasypy import Speakeasy

from data.graph_embeddings import GraphEmbeddings
from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


class ChatBot:
    def __init__(self, knowledge_graph: KnowledgeGraph, graph_embeddings: GraphEmbeddings, entity_extractor: NamedEntityRecognizer, relation_extractor: RelationExtractor):
        self.speakeasy = None
        self.rooms = []
        self.knowledge_graph = knowledge_graph
        self.graph_embeddings = graph_embeddings
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor

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
        print(f"\tProcessing message in room {room.my_alias}")
        message_string = message_object.message
        if self.is_sparql_query(message_string):
            try:
                query_result = self.knowledge_graph.custom_sparql_query(message_string)
                self.send_message(f"I found the following query result: \n{query_result}.", room)
                room.mark_as_processed(message_object)
                return
            except pyparsing.exceptions.ParseException:
                pass
        entities = self.entity_extractor.extract_entities(message_string)
        entity_uris = self.knowledge_graph.match_multiple_entities(entities)
        predicates = self.relation_extractor.extract_relations(message_string, entities)
        predicates = self.knowledge_graph.extract_best_matches(predicates)
        predicate_uris = self.knowledge_graph.match_multiple_predicates(predicates)

        print(f"\tFound the following entities: \n\t{entities}\n\t{entity_uris}")
        print(f"\tFound the following predicates: \n\t{predicates}\n\t{predicate_uris}")

        query_results = []
        for entity_uri in entity_uris:
            for predicate_uri in predicate_uris:
                query_results.append(self.knowledge_graph.query_graph(entity_uri, predicate_uri, False))

        embedding_results = []
        for entity in entities:
            for predicate_uri in predicate_uris:
                embedding_results.append(self.graph_embeddings.find_related_entities(entity, predicate_uri))

        flat_query_results = [item for sublist in query_results for item in sublist] if query_results else []
        flat_embedding_results = [item for sublist in embedding_results for item in sublist] if embedding_results else []

        if flat_query_results:
            query_response = "I found the following in the knowledge graph: " + ", ".join(flat_query_results)
        else:
            query_response = "I found nothing in the knowledge graph"

        if flat_embedding_results:
            embedding_response = "I found the following through embeddings: " + ", ".join(flat_embedding_results)
        else:
            embedding_response = "I found nothing through embeddings"

        response = f"{query_response}. {embedding_response}."
        self.send_message(response, room)
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
