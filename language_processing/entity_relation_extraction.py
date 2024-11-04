import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


class NamedEntityRecognizer:
    def __init__(self, model_name="dslim/bert-base-NER"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

    def extract_entities(self, question):
        entities = self.pipeline(question)
        if not entities:
            print("No entities found.")
            return []

        start = entities[0].get('start', 0)
        end = entities[-1].get('end', len(question))
        extracted_span = question[start:end].strip()

        return [extracted_span]


class RelationExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def extract_relations(self, question, entities):
        for entity in entities:
            question = question.replace(entity, "")

        doc = self.nlp(question)
        possible_relations = [token.text for token in doc if token.pos_ in ['VERB', 'NOUN', 'PROPN']]

        return possible_relations