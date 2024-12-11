import json
import logging

import numpy as np
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
from sklearn.metrics import pairwise_distances
import pickle
from thefuzz import process


class KnowledgeGraph(Graph):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("knowledge_graph")
        self.logger.info("Setting up knowledge graph...")
        self.parse("data/14_graph.nt", format="turtle")

        self.entity_embeddings = np.load("data/entity_embeds.npy")
        self._entity_to_id = pickle.load(open("data/entity_to_id.pkl", "rb"))
        self._id_to_entity = {id: entity for entity, id in self._entity_to_id.items()}

        self.movie_embeddings = np.load("data/movie_embeds.npy")
        self._movie_to_id = pickle.load(open("data/movie_to_id.pkl", "rb"))
        self._id_to_movie = {id: movie for movie, id in self._movie_to_id.items()}

        self.relation_embeddings = np.load("data/relation_embeds.npy")
        self._relation_to_id = pickle.load(open("data/relation_to_id.pkl", "rb"))
        self._id_to_relation = {id: relation for relation, id in self._relation_to_id.items()}

        self._relation_to_uri = pickle.load(open("data/relation_to_uri.pkl", "rb"))
        self._uri_to_relation = {}
        for relation, uris in self._relation_to_uri.items():
            for uri in uris:
                self._uri_to_relation[uri] = relation

        self._entity_to_uri = pickle.load(open("data/entity_to_uri.pkl", "rb"))
        self._uri_to_entity = {uri: entity for entity, uri in self._entity_to_uri.items()}

        with open("data/entity_to_uri.pkl", "wb") as f:
            pickle.dump(self._entity_to_uri, f)

        with open('data/images.json') as f:
            self.images_json = json.load(f)

        self.logger.info("Finished setting up knowledge graph.")

    def execute_sparql_query(self, query):
        query_result = [str(s) for s, in self.query(query)]
        return query_result

    def match_multiple_entities(self, entities):
        uris = []
        for entity in entities:
            uris += self.match_entity(entity)
        return uris

    def match_entity(self, entity):
        matched_entities = process.extract(entity, list(self._entity_to_uri.keys()), limit=2)
        self.logger.debug(f"Matched entities to '{entity}': {matched_entities}")
        return [self._entity_to_uri[entity] for entity, _ in matched_entities]

    def match_multiple_relations(self, relations):
        uris = []
        for relation in relations:
            uris += self.match_relation(relation)
        return uris

    def match_relation(self, relation):
        matched_relations = process.extract(relation, list(self._relation_to_uri.keys()), limit=2)
        self.logger.debug(f"Matched relations to '{relation}': {matched_relations}")
        return [self._relation_to_uri[entity] for entity, _ in matched_relations]

    def find_recommended_movies(self, entity_uris, top_n=5, top_n_per_movie=10):
        similar_movies_list = []
        for entity_uri in entity_uris:
            ent_id = self._entity_to_id.get(rdflib.term.URIRef(entity_uri))
            if ent_id is None:
                continue
            similar_movies = \
                self.get_similar_entities(self.entity_embeddings[ent_id], self.movie_embeddings, self._id_to_movie,
                                          top_n_per_movie)[["Entity", "Label", "Score"]]
            similar_movies["Count"] = 0
            similar_movies = similar_movies[similar_movies["Score"] != 0]
            similar_movies_list.append(similar_movies)
        if not similar_movies_list:
            return []
        all_similar_movies = pd.concat(similar_movies_list, ignore_index=True)

        # Remove original entities
        entity_ids = [uri.split('/')[-1] for uri in entity_uris]
        all_similar_movies = all_similar_movies[~all_similar_movies["Entity"].isin(entity_ids)]

        # Count how many times the same movie appears
        label_counts = all_similar_movies["Label"].value_counts()
        all_similar_movies["Count"] = all_similar_movies["Label"].map(label_counts)

        all_similar_movies = (all_similar_movies
                              .loc[all_similar_movies.groupby("Label")[
            "Score"].idxmin()]  # remove duplicates, only keep the ones with lowest score
                              .sort_values(by=["Count", "Score"],
                                           ascending=[False, True])  # sort by count first, then score
                              .head(top_n)  # only take the top n movies
                              )

        return all_similar_movies["Label"].tolist()

    def query_graph(self, entity_uri, relation_uri, obj=True):
        rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")

        if obj:
            query = f"SELECT DISTINCT ?x WHERE {{ ?x <{relation_uri}> <{entity_uri}>. }}"
        else:
            query = f"SELECT DISTINCT ?x WHERE {{ <{entity_uri}> <{relation_uri}> ?x. }}"

        self.logger.debug(f"Executing query: {query}")
        results = self.query(query)

        result_labels = []
        for row in results:
            if isinstance(row.x, Literal):
                result_labels.append(str(row.x))
            else:
                label_query = f"SELECT DISTINCT ?label WHERE {{ <{row.x}> <{rdfs.label}> ?label. }}"
                label_results = self.query(label_query)
                result_labels.extend([str(label_row.label) for label_row in label_results])

        self.logger.debug(f"Results: {result_labels}")
        return result_labels

    def get_similar_entities(self, entity_embedding, embeddings, id_to_embedding, top_n=50):
        """Finds the top_n most similar entities to the given entity embedding using the given embeddings and returns a
        dataframe with the entity, label, score and rank. A lower score means the entity is more similar to the given entity."""
        dist = pairwise_distances(entity_embedding.reshape(1, -1), embeddings).reshape(-1)
        most_likely = dist.argsort()

        similar_entities = pd.DataFrame([
            (
                id_to_embedding[id][len(rdflib.Namespace("http://www.wikidata.org/entity/")):],
                self.get_entity_label(str(id_to_embedding[id])),
                dist[id],
                rank + 1
            )
            for rank, id in enumerate(most_likely[:top_n])],
            columns=("Entity", "Label", "Score", "Rank")
        )
        return similar_entities

    def find_related_entities(self, entity_uri, relation_uri, top_n=1):
        """Finds the entities that are related to the given entity using the given relation and returns a list of the
        top n best matches as labels."""
        ent_id = self._entity_to_id.get(rdflib.term.URIRef(entity_uri))
        rel_id = self._relation_to_id.get(rdflib.term.URIRef(relation_uri))
        if ent_id is None or rel_id is None:
            return []

        result = self.entity_embeddings[ent_id] + self.relation_embeddings[rel_id]
        related_entities = self.get_similar_entities(result, self.entity_embeddings, self._id_to_entity, top_n)

        return related_entities["Label"].tolist()

    def get_imdb_id_from_entity(self, entity_uri):
        query = f"""
        SELECT ?imdbID WHERE {{
            <{entity_uri}> <http://www.wikidata.org/prop/direct/P345> ?imdbID .
        }}
        """
        results = self.execute_sparql_query(query)
        return results[0] if results else None

    def get_photos_from_imdb_id(self, imdb_id):
        for element in self.images_json:
            for cast_element in element["cast"]:
                if str(cast_element) == str(imdb_id):
                    photo = f"image:{element["img"].strip(".jpg")}"
                    return photo

    def get_entity_label(self, entity_uri):
        try:
            return self._uri_to_entity[entity_uri]
        except KeyError:
            return ""

    def get_relation_label(self, relation_uri):
        try:
            return self._uri_to_relation[relation_uri]
        except KeyError:
            return ""
