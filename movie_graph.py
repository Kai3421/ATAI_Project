import pyparsing
from rdflib import Graph
import pickle

class MovieGraph(Graph):
    def __init__(self, source):
        super().__init__()
        print("Setting up movie graph...")
        self.parse(source, format="turtle")
        print("Finished movie graph setup.")

    def custom_sparql_query(self, query):
        query_result = [str(s) for s, in self.query(query)]
        return query_result

    def saved_predicates(self, path):
        assert path.endswith(".pkl")
        print("Saving predicates...")
        predicates = list(self.predicates())
        with open(path, "wb") as f:
            pickle.dump(predicates, f)
        print(f"Finished saving predicates at {path}")
