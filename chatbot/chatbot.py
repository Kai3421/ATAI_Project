import logging
import re
from typing import Iterable
from pyparsing import ParseException


class ChatBot:
    def __init__(self, knowledge_graph, entity_extractor, relation_extractor, crowd_data):
        self.logger = logging.getLogger("chatbot")
        self.speakeasy = None
        self.rooms = []
        self.knowledge_graph = knowledge_graph
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.crowd_data = crowd_data

    def respond_to(self, message):
        response = None

        if self.is_sparql_query(message):
            try:
                response = self.execute_plain_sparql_query(message)
                self.logger.debug(f"Successfully executed SPARQL query.")
            except ParseException:
                self.logger.debug(f"Could not parse {message}")
        if response is None:
            if self.is_request_for_recommendation(message):
                response = self.make_recommendation(message)
            elif self.is_multimedia_question(message):
                response = self.make_multimedia_response(message)
            else:
                response = self.try_to_answer_question(message)

        return response

    def is_sparql_query(self, message):
        """Checks if the message might be a plain sparql query"""
        sparql_keywords = ["SELECT", "ASK", "WHERE", "PREFIX", "DESCRIBE", "CONSTRUCT"]
        has_keyword = any(keyword in message.upper() for keyword in sparql_keywords)
        has_sparql_symbols = bool(re.search(r"\?|<.*?>|\{.*?\}", message))
        return has_keyword and has_sparql_symbols

    def execute_plain_sparql_query(self, message):
        query_result = self.knowledge_graph.execute_sparql_query(message)
        response = f"I found the following query result: \n{query_result}."
        return response

    def is_request_for_recommendation(self, message):
        keywords = ["recommend", "suggest", "movies like", "should I watch", "similar", "any movies", "I like",
                    "I enjoy", "other movies", "I'm into"]

        for keyword in keywords:
            if keyword.lower() in message.lower():
                return True

        return False

    def make_recommendation(self, message):
        _, entity_uris = self.get_entities_and_uris(message)
        results = self.unique_flatten(self.knowledge_graph.find_recommended_movies(entity_uris))[:3]
        response = "I would recommend the following movies: " + ", ".join(results) + "."
        return response

    def is_multimedia_question(self, message):
        keywords = ["show me", "picture", "look like", "image"]

        for keyword in keywords:
            if keyword.lower() in message.lower():
                return True

        return False

    def make_multimedia_response(self, message):
        entities, entity_uris, relations, relation_uris = self.get_entity_and_relation_uris(message)
        image = "image:3640/rm215128064"
        response = f"I found the following images: \n{image}"
        return response

    def try_to_answer_question(self, message):
        entities, entity_uris, relations, relation_uris = self.get_entity_and_relation_uris(message)
        try:
            query_results = []
            embedding_results = []
            crowd_results = []
            for entity_uri in entity_uris:
                for relation_uri in relation_uris:
                    query_results.append(self.knowledge_graph.query_graph(entity_uri, relation_uri, False))
                    query_results.append(self.knowledge_graph.query_graph(entity_uri, relation_uri, True))
                    embedding_results.append(self.knowledge_graph.find_related_entities(entity_uri, relation_uri))
                    crowd_result = self.crowd_data.get_result(entity_uri, relation_uri)
                    if crowd_result is not None:
                        crowd_results.append(crowd_result)

            flat_query_results = self.unique_flatten(query_results)[:3]
            flat_embedding_results = self.unique_flatten(embedding_results)[:3]

            if flat_query_results:
                query_response = "I found the following in the knowledge graph: " + ", ".join(flat_query_results) + "."
            else:
                query_response = "I found nothing in the knowledge graph."

            if flat_embedding_results:
                embedding_response = "I found the following through embeddings: " + ", ".join(flat_embedding_results) + "."
            else:
                embedding_response = "I found nothing through embeddings."

            if crowd_results:
                crowd_response = "Crowd sourcing suggests that the answer is "
                for result in crowd_results:
                    object, inter_rater, votes = result
                    crowd_response += f"{object} with inter-rater agreement {inter_rater:.3f} and {votes}. "
            else:
                crowd_response = "I found nothing through crowd sourcing."

            response = f"{query_response} \n\n{embedding_response} \n\n{crowd_response}"
        except ParseException as error:
            response = f"Something went wrong with the query, please try again."
            self.logger.error(error)
            self.logger.error(f"User message: {message}")

        return response

    def get_entities_and_uris(self, message):
        entities = self.unique_flatten(self.entity_extractor.extract_entities(message))
        entity_uris = self.unique_flatten(self.knowledge_graph.match_multiple_entities(entities))
        return entities, entity_uris

    def get_entity_and_relation_uris(self, message):
        entities = self.entity_extractor.extract_entities(message)
        merged_entities = " ".join(entities)
        entity_uris = self.unique_flatten(self.knowledge_graph.match_entity(merged_entities))
        relations = self.relation_extractor.extract_relations(message, entities)
        merged_relations = " ".join(relations)
        relation_uris = self.unique_flatten(self.knowledge_graph.match_relation(merged_relations))
        return merged_entities, entity_uris, merged_relations, relation_uris

    def flatten(self, nested_list):
        """Recursive generator to flatten nested iterables (like lists, tuples, sets)."""
        for x in nested_list:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from self.flatten(x)
            else:
                yield x

    def unique_flatten(self, nested_list):
        """Takes a list and returns a flattened list with all duplicate elements removed."""
        res = list(set(self.flatten(nested_list))) if nested_list else []
        return res
