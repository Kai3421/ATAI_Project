import os.path
from chatbot import ChatBot
from movie_graph import MovieGraph
import pickle

if __name__ == "__main__":
    knowledge_graph = MovieGraph("dataset/14_graph.nt")
    predicates_path = "predicates.pkl"
    if not os.path.isfile(predicates_path):
        knowledge_graph.saved_predicates("predicates.pkl")

    chatbot = ChatBot(knowledge_graph)
    host = "https://speakeasy.ifi.uzh.ch"
    username = "playful-panther"
    with open("password.txt") as file:
        password = file.readline()
    chatbot.login(host=host, username=username, password=password)
    chatbot.run()