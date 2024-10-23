import pyparsing
from speakeasypy import Speakeasy
import rdflib

print("Starting bot...")
graph = rdflib.Graph()
graph.parse('dataset/14_graph.nt', format='turtle')
print("Graph created.")

with open('password.txt') as file:
    password = file.readline()
speakeasy = Speakeasy(host='https://speakeasy.ifi.uzh.ch', username='playful-panther', password=password)
speakeasy.login()

while True:
    rooms = speakeasy.get_rooms(active=True)
    for room in rooms:
        if not room.initiated:
            room.post_messages(f"Hi!, I'm playful panther. Please post a SPARQL query!")
            room.initiated = True
        for message in room.get_messages(only_partner=True, only_new=True):
            room.post_messages(f"Let me look that up for you...")
            try:
                query_result = [str(s) for s, in graph.query(message.message)]
                room.post_messages(str(query_result))
            except pyparsing.exceptions.ParseException:
                room.post_messages(f"Sorry, that seems to be an invalid query! Please try again.")
            room.mark_as_processed(message)
        for reaction in room.get_reactions(only_new=True):
            room.post_messages(f"Received your reaction: '{reaction.type}' ")
            room.mark_as_processed(reaction)