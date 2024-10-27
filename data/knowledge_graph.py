import editdistance
from rdflib import Graph, Namespace, URIRef
import pickle

class KnowledgeGraph(Graph):
    def __init__(self, source):
        super().__init__()
        print("Setting up knowledge graph...")
        self.parse(source, format="turtle")
        self.predicate_map = {}
        self.node_map = {}
        self._initialize_nodes_and_predicates()
        print("Finished movie graph setup.")

    def _initialize_nodes_and_predicates(self):
        rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        for node in self.all_nodes():
            if isinstance(node, URIRef) and self.value(node, rdfs.label):
                self.node_map[node.toPython()] = self.value(node, rdfs.label).toPython()

        for s, p, o in self:
            if isinstance(p, URIRef) and self.value(p, rdfs.label):
                self.predicate_map[p.toPython()] = self.value(p, rdfs.label).toPython()

    def match_multiple_entities(self, entities):
        uris = []
        for entity in entities:
            uris.append(self.match_entity(entity))
        return uris

    def match_entity(self, entity):
        min_distance = float('inf')
        matched_entity_uri = None
        for uri, label in self.node_map.items():
            distance = editdistance.eval(entity, label)
            if distance < min_distance:
                min_distance = distance
                matched_entity_uri = uri
        return matched_entity_uri

    def match_multiple_predicates(self, relations):
        uris = []
        for relation in relations:
            uris.append(self.match_predicate(relation))
        return uris

    def match_predicate(self, relation):
        min_distance = float('inf')
        matched_predicate_uri = None
        for uri, label in self.predicate_map.items():
            distance = editdistance.eval(relation, label)
            if distance < min_distance:
                min_distance = distance
                matched_predicate_uri = uri
        return matched_predicate_uri

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

    def custom_sparql_query(self, query):
        query_result = [str(s) for s, in self.query(query)]
        return query_result

    # def save_predicates(self, path):
    #     assert path.endswith(".pkl")
    #     print("Saving predicates...")
    #     predicates = list(self.predicates())
    #     with open(path, "wb") as f:
    #         pickle.dump(predicates, f)
    #     print(f"Finished saving predicates at {path}")


