import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import editdistance


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
    def __init__(self, predicates):
        self.predicates = predicates
        self.nlp = spacy.load('en_core_web_sm')

    def extract_relations(self, question, entities):
        for entity in entities:
            question = question.replace(entity, "")

        doc = self.nlp(question)
        possible_relations = [token.text for token in doc if token.pos_ in ['VERB', 'NOUN', 'PROPN']]

        return self.compute_similarity(possible_relations)

    def compute_similarity(self, relations):
        similarity_results = {}
        for relation in relations:
            best_match = None
            min_distance = float('inf')
            for predicate in self.predicates:
                distance = editdistance.eval(relation, predicate)
                if distance < min_distance:
                    min_distance = distance
                    best_match = predicate
            similarity_results[relation] = {'best_match': best_match, 'edit_distance': min_distance}
        return similarity_results