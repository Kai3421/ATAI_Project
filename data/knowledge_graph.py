import logging
import csv
import json

import editdistance
import numpy as np
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, URIRef, Literal

from sklearn.metrics import pairwise_distances


class KnowledgeGraph(Graph):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("knowledge_graph")
        self.logger.info("Setting up knowledge graph...")
        self.parse("data/14_graph.nt", format="turtle")

        self.entity_embeddings = np.load("data/entity_embeds.npy")
        self.movie_embeddings = np.load("data/movie_embeds.npy")
        self.relation_embeddings = np.load("data/relation_embeds.npy")
        self._entity_to_id = self._load_ids("data/entity_ids.del")
        self._id_to_entity = {v: k for k, v in self._entity_to_id.items()}
        self._movie_to_id = self._load_ids("data/movie_ids.del")
        self._id_to_movie = {v: k for k, v in self._movie_to_id.items()}
        self._relation_to_id = self._load_ids("data/relation_ids.del")
        self._id_to_relation = {v: k for k, v in self._relation_to_id.items()}

        self.relation_uri_map = {}
        self.entity_uri_map = {}
        self._initialize_entity_and_relation_maps()

        self.logger.info("Finished setting up knowledge graph.")

    @staticmethod
    def _load_ids(path):
        with open(path, 'r') as ifile:
            return {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}

    def _initialize_entity_and_relation_maps(self):
        rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        for node in self.all_nodes():
            if isinstance(node, URIRef) and self.value(node, rdfs.label):
                self.entity_uri_map[node.toPython()] = self.value(node, rdfs.label).toPython()

        with open("data/predicate_aliases.json", "r") as json_file:
            self.relation_uri_map = json.load(json_file)

    def get_entity_label(self, entity_uri):
        try:
            return self.entity_uri_map[entity_uri]
        except KeyError:
            return ""

    def match_multiple_entities(self, entities):
        uris = []
        for entity in entities:
            uris += self.match_entity(entity)
        return uris

    def match_entity(self, entity):
        min_distance = float('inf')
        matched_entity_uris = []

        for uri, label in self.entity_uri_map.items():
            distance = editdistance.eval(entity, label)
            if distance < min_distance:
                min_distance = distance
                matched_entity_uris = [uri]
            elif distance == min_distance:
                matched_entity_uris.append(uri)
        self.logger.debug(f"Entity: '{entity}', URIs: {matched_entity_uris}")
        return matched_entity_uris

    def match_multiple_predicates(self, relations):
        uris = []
        for relation in relations:
            uris += self.match_predicate(relation)
        return uris

    def match_predicate(self, relation):
        min_distance = float('inf')
        matched_predicate_uris = []

        for label, uris in self.relation_uri_map.items():
            if not isinstance(uris, list):
                uris = [uris]

            distance = editdistance.eval(relation, label)

            if distance < min_distance:
                min_distance = distance
                matched_predicate_uris = uris
            elif distance == min_distance:
                matched_predicate_uris.extend(uris)
        self.logger.debug(f"Relation: {relation}, URIs: {matched_predicate_uris}")
        return matched_predicate_uris

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

    def find_recommended_movies(self, entity_uris, top_n=5, top_n_per_movie=10):
        similar_movies_list = []
        for entity_uri in entity_uris:
            ent_id = self._entity_to_id.get(rdflib.term.URIRef(entity_uri))
            if ent_id is None:
                continue
            similar_movies = self.get_similar_entities(self.entity_embeddings[ent_id], self.movie_embeddings, self._id_to_movie, top_n_per_movie)[["Entity", "Label", "Score"]]
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
                              .loc[all_similar_movies.groupby("Label")["Score"].idxmin()] # remove duplicates, only keep the ones with lowest score
                              .sort_values(by=["Count", "Score"], ascending=[False, True]) # sort by count first, then score
                              .head(top_n) # only take the top n movies
                              )

        return all_similar_movies["Label"].tolist()

    def custom_sparql_query(self, query):
        query_result = [str(s) for s, in self.query(query)]
        return query_result
