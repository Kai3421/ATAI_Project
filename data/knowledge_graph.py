import csv

import editdistance
import numpy as np
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, URIRef

from sklearn.metrics import pairwise_distances


class KnowledgeGraph(Graph):
    def __init__(self):
        super().__init__()
        print("Setting up knowledge graph...")
        self.parse("data/14_graph.nt", format="turtle")

        self.entity_embeddings = np.load("data/entity_embeds.npy")
        self.relation_embeddings = np.load("data/relation_embeds.npy")
        self._entity_to_id = self._load_ids("data/entity_ids.del")
        self._id_to_entity = {v: k for k, v in self._entity_to_id.items()}
        self._relation_to_id = self._load_ids("data/relation_ids.del")
        self._id_to_relation = {v: k for k, v in self._relation_to_id.items()}

        self.relation_uri_map = {}
        self.entity_uri_map = {}
        self._initialize_entity_and_relation_maps()

        print("Finished knowledge graph setup.")

    @staticmethod
    def _load_ids(path):
        with open(path, 'r') as ifile:
            return {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}

    def _initialize_entity_and_relation_maps(self):
        rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        for node in self.all_nodes():
            if isinstance(node, URIRef) and self.value(node, rdfs.label):
                self.entity_uri_map[node.toPython()] = self.value(node, rdfs.label).toPython()

        for s, p, o in self:
            if isinstance(p, URIRef) and self.value(p, rdfs.label):
                self.relation_uri_map[p.toPython()] = self.value(p, rdfs.label).toPython()

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
        return matched_entity_uris

    def match_multiple_predicates(self, relations):
        uris = []
        for relation in relations:
            uris += self.match_predicate(relation)
        return uris

    def match_predicate(self, relation):
        min_distance = float('inf')
        matched_predicate_uris = []
        for uri, label in self.relation_uri_map.items():
            distance = editdistance.eval(relation, label)
            if distance < min_distance:
                min_distance = distance
                matched_predicate_uris = [uri]
            elif distance == min_distance:
                matched_predicate_uris.append(uri)
        return matched_predicate_uris

    @staticmethod
    def extract_best_matches(data):
        return [info['best_match'] for info in data.values()]

    def query_graph(self, entity_uri, relation_uri, obj=True):
        rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        query = (
            f"SELECT DISTINCT ?x ?y WHERE {{ ?x <{relation_uri}> <{entity_uri}>. ?x <{rdfs.label}> ?y. }}"
            if obj else
            f"SELECT DISTINCT ?x ?y WHERE {{ <{entity_uri}> <{relation_uri}> ?x. ?x <{rdfs.label}> ?y. }}"
        )
        print(f"\tExecuting query: \n\t{query}\n")
        results = self.query(query)
        return [str(row.y) for row in results]

    def find_related_entities(self, entity: str, relation: str, top_n: int = 1):
        ent_id = self._entity_to_id.get(rdflib.term.URIRef(entity))
        print(f"\tFound id {ent_id} for {entity}")
        rel_id = self._relation_to_id.get(rdflib.term.URIRef(relation))
        print(f"\tFound id {rel_id} for {relation}")
        if ent_id is None or rel_id is None:
            return []

        lhs = self.entity_embeddings[ent_id] + self.relation_embeddings[rel_id]

        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_embeddings).reshape(-1)
        most_likely = dist.argsort()

        results = pd.DataFrame([
            (
                self._id_to_entity[idx][len(rdflib.Namespace('http://www.wikidata.org/entity/')):],
                self.entity_uri_map[str(self._id_to_entity[idx])],
                dist[idx],
                rank + 1
            )
            for rank, idx in enumerate(most_likely[:top_n])],
            columns=('Entity', 'Label', 'Score', 'Rank')
        )
        print(f"\t{results}")
        return results["Label"].tolist()

    def find_similar_entities(self, query_entity: str, top_n: int = 10):
        ent_id = self._entity_to_id.get(self.entity_uri_map.get(query_entity, None))
        if ent_id is None:
            return []

        dist = pairwise_distances(self.entity_embeddings[ent_id].reshape(1, -1), self.entity_embeddings).reshape(-1)
        most_likely = dist.argsort()

        results = pd.DataFrame([
            (
                self._id_to_entity[idx][len(rdflib.Namespace('http://www.wikidata.org/entity/')):],
                self.entity_uri_map[self._id_to_entity[idx]],
                dist[idx],
                rank + 1
            )
            for rank, idx in enumerate(most_likely[:top_n])],
            columns=('Entity', 'Label', 'Score', 'Rank')
        )
        print(f"\t{results}")
        return results

    def custom_sparql_query(self, query):
        query_result = [str(s) for s, in self.query(query)]
        return query_result
