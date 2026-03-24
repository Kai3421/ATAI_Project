# ATAI Knowledge Graph Chatbot
**UZH Advanced Topics in Artificial Intelligence** | Fall 2024

A conversational question-answering system over a structured knowledge graph, built as part of the UZH ATAI course. The system combines named entity recognition, relation extraction, crowd-sourced data integration, and a movie recommender to answer natural language queries.

## Architecture

```
User input
  └─> Named Entity Recognizer      (spaCy-based NER)
  └─> Relation Extractor           (extracts subject–predicate–object triples)
  └─> Knowledge Graph              (SPARQL-queryable RDF graph)
  └─> Crowd Data Module            (integrates crowd-sourced annotations)
  └─> ChatBot                      (orchestrates components, generates response)
  └─> Chatroom Manager             (handles multi-turn session state)
```

## Key components

| Module | Description |
|---|---|
|  | Core dialogue manager — routes queries to KG, crowd data, or recommender |
|  | Multi-turn session and state management |
|  | NER and relation extraction pipeline |
|  | RDF knowledge graph with SPARQL query interface |
|  | Crowd-sourced annotation loader and resolver |
|  | Collaborative filtering movie recommender (numpy embeddings) |

## Setup and usage

Install dependencies:
```bash
pip install numpy pandas matplotlib spacy
```

Run the chatbot:
```bash
python main.py
```

Run in a terminal (not a notebook) to accept interactive user input.

## Course context

Built for the UZH [Advanced Topics in Artificial Intelligence](https://www.ifi.uzh.ch/en/ddis/teaching/atai.html) course, which covers knowledge graphs, semantic web technologies, NLP pipelines, and conversational agents.
