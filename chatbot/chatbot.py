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

    def login(self, host, username, password):
        self.speakeasy = Speakeasy(host=host, username=username, password=password)
        self.speakeasy.login()

    def logout(self):
        self.speakeasy.logout()

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

        if is_sparql_query(message_string):
            try:
                query_result = self.knowledge_graph.custom_sparql_query(message_string)
                self.send_message(f"I found the following query result: \n{query_result}.", room)
                room.mark_as_processed(message_object)
            except ParseException:
                pass  # ignore invalid SPARQL query, try the usual way instead.
        elif "recommend" in message_string.lower():
            entities = unique_flatten(self.entity_extractor.extract_entities(message_string))
            entity_uris = unique_flatten(self.knowledge_graph.match_multiple_entities(entities))
            similar_entities = []
            for entity_uri in entity_uris:
                similar_entities.append(self.knowledge_graph.find_similar_entities(entity_uri))
            flat_results = unique_flatten(similar_entities)
            response = "I would recommend the following movies: " + ", ".join(flat_results) + "."
            self.send_message(response, room)
            room.mark_as_processed(message_object)
        else:
            entities = unique_flatten(self.entity_extractor.extract_entities(message_string))
            entity_uris = unique_flatten(self.knowledge_graph.match_multiple_entities(entities))
            predicates = unique_flatten(self.relation_extractor.extract_relations(message_string, entities))
            predicate_uris = unique_flatten(self.knowledge_graph.match_multiple_predicates(predicates))
            try:
                query_results = []
                embedding_results = []
                for entity_uri in entity_uris:
                    for predicate_uri in predicate_uris:
                        query_results.append(self.knowledge_graph.query_graph(entity_uri, predicate_uri, False))
                        query_results.append(self.knowledge_graph.query_graph(entity_uri, predicate_uri, True))
                        embedding_results.append(self.knowledge_graph.find_related_entities(entity_uri, predicate_uri))

                flat_query_results = unique_flatten(query_results)
                flat_embedding_results = unique_flatten(embedding_results)

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
            except ParseException as error:
                response = f"Something went wrong with the query, please try again."
                self.send_message(response, room)
                room.mark_as_processed(message_object)
                self.logger.error(error)
                self.logger.error(f"User message: {message_string}")


    def process_reaction(self, reaction, room):
        self.send_message(f"Received your reaction: '{reaction.type}' ", room)
        room.mark_as_processed(reaction)


    @staticmethod
    def send_message(message, room):
        room.post_messages(sanitize(message))


def is_sparql_query(message):
    sparql_keywords = ["SELECT", "ASK", "WHERE", "PREFIX", "DESCRIBE", "CONSTRUCT"]
    has_keyword = any(keyword in message.upper() for keyword in sparql_keywords)
    has_sparql_symbols = bool(re.search(r"\?|<.*?>|\{.*?\}", message))
    return has_keyword and has_sparql_symbols


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


def flatten(l):
    """Recursive generator to flatten nested iterables (like lists, tuples, sets)."""
    for x in l:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def unique_flatten(l):
    res = list(set(flatten(l))) if l else []
    return res