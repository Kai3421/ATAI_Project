import os.path
from chatbot.chatbot import ChatBot
from data.movie_graph import MovieGraph

if __name__ == "__main__":
    knowledge_graph = MovieGraph("data/14_graph.nt")
    predicates_path = "data/predicates.pkl"
    if not os.path.isfile(predicates_path):
        knowledge_graph.save_predicates("predicates.pkl")

    chatbot = ChatBot(knowledge_graph)
    host = "https://speakeasy.ifi.uzh.ch"
    username = "playful-panther"
    with open("password.txt") as file:
        password = file.readline()
    chatbot.login(host=host, username=username, password=password)
    chatbot.run()