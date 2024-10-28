from chatbot.chatbot import ChatBot
from data.knowledge_graph import KnowledgeGraph
from language_processing.entity_relation_extraction import NamedEntityRecognizer, RelationExtractor


if __name__ == "__main__":
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
