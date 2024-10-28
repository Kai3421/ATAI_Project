import os
import dill

from chatbot.chatbot import ChatBot
from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


if __name__ == "__main__":
    # saved_graph_path = "data/knowledge_graph.pkl"
    # if os.path.isfile(saved_graph_path):
    #     print(f"Found saved knowledge graph, loading...")
    #     with open(saved_graph_path, "rb") as f:
    #         knowledge_graph = dill.load(f)
    #     print(f"Loaded knowledge graph.")
    # else:
    #     print(f"Didn't find saved knowledge graph, creating one...")
    #     knowledge_graph = KnowledgeGraph()
    #     print(f"Saving knowledge graph...")
    #     with open(saved_graph_path, "wb") as f:
    #         dill.dump(knowledge_graph, f)
    #     print(f"Saved knowledge graph.")

    knowledge_graph = KnowledgeGraph()

    named_entity_recognizer = NamedEntityRecognizer()
    relation_extractor = RelationExtractor(list(knowledge_graph.relation_uri_map.values()))

    chatbot = ChatBot(knowledge_graph, named_entity_recognizer, relation_extractor)
    host = "https://speakeasy.ifi.uzh.ch"
    username = "playful-panther"
    with open("chatbot/password.txt") as file:
        password = file.readline()
    chatbot.login(host=host, username=username, password=password)
    chatbot.run()
