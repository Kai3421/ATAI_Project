import os.path
from chatbot.chatbot import ChatBot
from data.graph_embeddings import GraphEmbeddings
from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


if __name__ == "__main__":
    knowledge_graph = KnowledgeGraph("data/14_graph.nt")
    graph_embeddings = GraphEmbeddings(knowledge_graph, 'data/entity_embeds.npy', 'data/relation_embeds.npy', 'data/entity_ids.del', 'data/relation_ids.del')
    named_entity_recognizer = NamedEntityRecognizer()
    relation_extractor = RelationExtractor(list(knowledge_graph.predicate_map.values()))

    chatbot = ChatBot(knowledge_graph, graph_embeddings, named_entity_recognizer, relation_extractor)
    host = "https://speakeasy.ifi.uzh.ch"
    username = "playful-panther"
    with open("chatbot/password.txt") as file:
        password = file.readline()
    chatbot.login(host=host, username=username, password=password)
    chatbot.run()
