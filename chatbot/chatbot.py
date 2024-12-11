import logging
import re
from typing import Iterable
from pyparsing import ParseException
import random


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
        entities, entity_uris = self.get_entities_and_uris(message)
        results = self.unique_flatten(self.knowledge_graph.find_recommended_movies(entity_uris))[:3]
        # response = "I would recommend the following movies: " + ", ".join(results) + "."
        response = f"Based on " + ", ".join(entities) + " I would recommend: " + ", ".join(results) + "."
        return response

    def is_multimedia_question(self, message):
        keywords = ["show me", "picture", "look like", "image", "looks like", "photo"]

        for keyword in keywords:
            if keyword.lower() in message.lower():
                return True

        return False

    def make_multimedia_response(self, message):
        entity, entity_uris, _, _ = self.get_entity_and_relation_uris(message)
        for uri in entity_uris:
            imdb_id = self.knowledge_graph.get_imdb_id_from_entity(uri)
            if imdb_id:
                photo = self.knowledge_graph.get_photos_from_imdb_id(imdb_id)
                if photo:
                    # response = f"Here is a picture of {entity}: {photo}"
                    response = f"Here is an image of {entity}: \n{photo}"
                    return response

        return "No relevant photos found."

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
                        obj, inter_rater, votes = crowd_result
                        label = self.knowledge_graph.get_entity_label(obj)
                        if not label:
                            label = self.knowledge_graph.get_relation_label(obj)
                            if not label:
                                label = obj
                        crowd_results.append((label, inter_rater, votes))

            flat_query_results = self.unique_flatten(query_results)[:3]
            flat_embedding_results = self.unique_flatten(embedding_results)[:3]

            if flat_query_results:
                query_response = "Querying the knowledge graph, I found " + ", ".join(flat_query_results) + "."
            else:
                query_response = "I found nothing in the knowledge graph."

            if flat_embedding_results:
                # embedding_response = "\n\nI found the following through embeddings: " + ", ".join(flat_embedding_results) + "."
                embedding_response = "\n\nI think it is: " + ", ".join(
                    flat_embedding_results) + ". (Found through embeddings)"
            else:
                embedding_response = "\n\nI found nothing through embeddings."

            if crowd_results:
                crowd_response = "\n\nCrowd sourcing suggests that the answer is "
                for result in crowd_results:
                    obj, inter_rater, votes = result
                    crowd_response += f"{obj} with inter-rater agreement {inter_rater:.3f} and {votes}. "
            else:
                crowd_response = ""

            response = f"{query_response} {embedding_response} {crowd_response}"
        except ParseException as error:
            response = f"Something went wrong with the query, please try again."
            self.logger.error(error)
            self.logger.error(f"User message: {message}")

        return response

    def get_entities_and_uris(self, message):
        entities = self.unique_flatten(self.entity_extractor.extract_multiple_entities(message))
        entity_uris = self.unique_flatten(self.knowledge_graph.match_multiple_entities(entities))
        return entities, entity_uris

    def get_entity_and_relation_uris(self, message):
        entity = self.entity_extractor.extract_single_entity(message)
        if entity:
            entity_uris = self.unique_flatten(self.knowledge_graph.match_entity(entity))
        else:
            entity_uris = []
        relation = self.relation_extractor.extract_relations(message, [entity])
        if relation:
            relation_uris = self.unique_flatten(self.knowledge_graph.match_relation(relation))
        else:
            relation_uris = []
        return entity, entity_uris, relation, relation_uris

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
