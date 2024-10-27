import csv
import numpy as np
import pandas as pd
import rdflib
from sklearn.metrics import pairwise_distances


class GraphEmbeddings:
    def __init__(self, graph, entity_embeds_path: str, relation_embeds_path: str, entity_ids_path: str, relation_ids_path: str):
        self.graph = graph
        self.entity_embeddings = np.load(entity_embeds_path)
        self.relation_embeddings = np.load(relation_embeds_path)
        self.entity_to_id = self.load_ids(entity_ids_path)
        self.relation_to_id = self.load_ids(relation_ids_path)
        self.id_to_entity = {v: k for k, v in self.entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        self.entity_to_label = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(rdflib.namespace.RDFS.label)}
        self.label_to_entity = {lbl: ent for ent, lbl in self.entity_to_label.items()}
        self.relation_to_label = graph.predicate_map
        self.label_to_relation = {str(lbl): uri for uri, lbl in self.relation_to_label.items()}

    def load_ids(self, path):
        with open(path, 'r') as ifile:
            return {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}

    def find_similar_entities(self, query_entity: str, top_n: int = 10):
        ent_id = self.entity_to_id.get(self.label_to_entity.get(query_entity, None))
        if ent_id is None:
            return []

        dist = pairwise_distances(self.entity_embeddings[ent_id].reshape(1, -1), self.entity_embeddings).reshape(-1)
        most_likely = dist.argsort()

        results = pd.DataFrame([
            (
                self.id_to_entity[idx][len(rdflib.Namespace('http://www.wikidata.org/entity/')):],
                self.entity_to_label[self.id_to_entity[idx]],
                dist[idx],
                rank + 1
            )
            for rank, idx in enumerate(most_likely[:top_n])],
            columns=('Entity', 'Label', 'Score', 'Rank')
        )
        return results

    def find_related_entities(self, query_entity: str, relation: str, top_n: int = 1):
        ent_id = self.entity_to_id.get(self.label_to_entity.get(query_entity, None))
        rel_id = self.relation_to_id.get(rdflib.term.URIRef(relation))
        if ent_id is None or rel_id is None:
            return []

        lhs = self.entity_embeddings[ent_id] + self.relation_embeddings[rel_id]

        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_embeddings).reshape(-1)
        most_likely = dist.argsort()
        print(f"\t{most_likely}")

        results = pd.DataFrame([
            (
                self.id_to_entity[idx][len(rdflib.Namespace('http://www.wikidata.org/entity/')):],
                self.entity_to_label[self.id_to_entity[idx]],
                dist[idx],
                rank + 1
            )
            for rank, idx in enumerate(most_likely[:top_n])],
            columns=('Entity', 'Label', 'Score', 'Rank')
        )
        return results["Label"].tolist()
