import logging
import re
from typing import Iterable

import pyparsing
from pyparsing import ParseException
from speakeasypy import Speakeasy

from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


class ChatBot:
    def __init__(self, knowledge_graph: KnowledgeGraph, entity_extractor: NamedEntityRecognizer, relation_extractor: RelationExtractor):
        self.logger = logging.getLogger("chatbot")
        self.speakeasy = None
        self.rooms = []
        self.knowledge_graph = knowledge_graph
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.login()

    def login(self):
        host = "https://speakeasy.ifi.uzh.ch"
        username = "playful-panther"
        with open("chatbot/password.txt") as file:
            password = file.readline()
        self.speakeasy = Speakeasy(host=host, username=username, password=password)
        self.speakeasy.login()

    def run(self):
        """Iterate over all the rooms and handle messages and reactions."""
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
        self.logger.debug(f"Processing message in room {room.my_alias}:")
        message_string = message_object.message
        self.logger.debug(f"Message: {message_string}")
        response = None

        if is_sparql_query(message_string):
            try:
                response = self.execute_plain_sparql_query(message_string)
                self.logger.debug(f"Successfully executed SPARQL query.")
            except ParseException:
                self.logger.debug(f"Could not parse {message_string}")
        if response is None:
            if is_request_for_recommendation(message_string):
                response = self.make_recommendation(message_string)
            else:
                response = self.try_to_answer_question(message_string)

        self.logger.debug(f"Response: {response}")
        self.send_message(response, room)
        room.mark_as_processed(message_object)

    def execute_plain_sparql_query(self, message_string):
        query_result = self.knowledge_graph.custom_sparql_query(message_string)
        response = f"I found the following query result: \n{query_result}."
        return response

    def make_recommendation(self, message_string):
        _, entity_uris = self.get_entities_and_uris(message_string)
        results = unique_flatten(self.knowledge_graph.find_recommended_movies(entity_uris))[:3]
        response = "I would recommend the following movies: " + ", ".join(results) + "."
        return response

    def try_to_answer_question(self, message_string):
        entities, entity_uris = self.get_entities_and_uris(message_string)
        _, predicate_uris = self.get_predicates_and_uris(message_string, entities)
        try:
            query_results = []
            embedding_results = []
            for entity_uri in entity_uris:
                for predicate_uri in predicate_uris:
                    query_results.append(self.knowledge_graph.query_graph(entity_uri, predicate_uri, False))
                    query_results.append(self.knowledge_graph.query_graph(entity_uri, predicate_uri, True))
                    embedding_results.append(self.knowledge_graph.find_related_entities(entity_uri, predicate_uri))

            flat_query_results = unique_flatten(query_results)[:3]
            flat_embedding_results = unique_flatten(embedding_results)[:3]

            if flat_query_results:
                query_response = "I found the following in the knowledge graph: " + ", ".join(flat_query_results)
            else:
                query_response = "I found nothing in the knowledge graph"

            if flat_embedding_results:
                embedding_response = "I found the following through embeddings: " + ", ".join(flat_embedding_results)
            else:
                embedding_response = "I found nothing through embeddings"

            response = f"{query_response}. {embedding_response}."
        except ParseException as error:
            response = f"Something went wrong with the query, please try again."
            self.logger.error(error)
            self.logger.error(f"User message: {message_string}")

        return response

    def get_entities_and_uris(self, message_string):
        entities = unique_flatten(self.entity_extractor.extract_entities(message_string))
        entity_uris = unique_flatten(self.knowledge_graph.match_multiple_entities(entities))
        return entities, entity_uris

    def get_predicates_and_uris(self, message_string, entities):
        predicates = unique_flatten(self.relation_extractor.extract_relations(message_string, entities))
        predicate_uris = unique_flatten(self.knowledge_graph.match_multiple_predicates(predicates))
        return predicates, predicate_uris

    def process_reaction(self, reaction, room):
        self.send_message(f"Received your reaction: '{reaction.type}' ", room)
        room.mark_as_processed(reaction)

    @staticmethod
    def send_message(message, room):
        """Sanitizes the message and then sends it to the room."""
        room.post_messages(sanitize(message))

    def clear(self):
        self.rooms = self.speakeasy.get_rooms()
        for room in self.rooms:
            for message in room.get_messages(only_partner=True, only_new=True):
                room.mark_as_processed(message)

            for reaction in room.get_reactions(only_new=True):
                room.mark_as_processed(reaction)


    def logout(self):
        self.speakeasy.logout()


def is_sparql_query(message):
    """Checks if the message might be a plain sparql query"""
    sparql_keywords = ["SELECT", "ASK", "WHERE", "PREFIX", "DESCRIBE", "CONSTRUCT"]
    has_keyword = any(keyword in message.upper() for keyword in sparql_keywords)
    has_sparql_symbols = bool(re.search(r"\?|<.*?>|\{.*?\}", message))
    return has_keyword and has_sparql_symbols


def is_request_for_recommendation(message_string):
    keywords = {"recommend", "suggest", "movies like", "should I watch", "similar", "any movies", "I like", "I enjoy",
                "other movies", "I'm into"}

    for keyword in keywords:
        if keyword.lower() in message_string.lower():
            return True

    return False


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


def flatten(nested_list):
    """Recursive generator to flatten nested iterables (like lists, tuples, sets)."""
    for x in nested_list:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def unique_flatten(nested_list):
    """Takes a list and returns a flattened list with all duplicate elements removed."""
    res = list(set(flatten(nested_list))) if nested_list else []
    return res