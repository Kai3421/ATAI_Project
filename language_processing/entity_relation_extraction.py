import logging
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

class NamedEntityRecognizer:
    def __init__(self, model_name="dslim/bert-base-NER"):
        self.logger = logging.getLogger("named_entity_recognizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple", device="cpu")


    def extract_entities(self, question):
        entities = self.pipeline(question)
        if not entities:
            self.logger.debug("No entities found.")
            return []

        start = entities[0].get('start', 0)
        end = entities[-1].get('end', len(question))
        extracted_span = question[start:end].strip()

        self.logger.debug(f"Extracted entities: {extracted_span}")

        return [extracted_span]


class RelationExtractor:
    def __init__(self):
        self.logger = logging.getLogger("relation_extractor")
        self.nlp = spacy.load('en_core_web_sm')

    def extract_relations(self, question, entities):
        for entity in entities:
            question = question.replace(entity, "")

        doc = self.nlp(question)
        possible_relations = [token.text for token in doc if token.pos_ in ['VERB', 'NOUN', 'PROPN']]

        self.logger.debug(f"Extracted relations: {possible_relations}")

        return possible_relations