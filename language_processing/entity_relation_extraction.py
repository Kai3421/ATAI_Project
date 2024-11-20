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
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="average", device="cpu")

    def extract_entities(self, question):
        found_entities = self.pipeline(question)
        self.logger.debug(f"Extracted entities: {found_entities}")
        if not found_entities:
            self.logger.debug("No entities found.")
            return []

        entities = [question[item['start']:item['end']] for item in found_entities]
        entities = self.split_at_commas(entities)
        entities = self.combine_articles(entities)
        entities = self.capitalize(entities)

        return entities

    def split_at_commas(self, entities):
        result = []
        for entity in entities:
            split_parts = [part.strip() for part in entity.split(',')]
            result.extend(split_parts)
        return result

    def combine_articles(self, entities):
        articles = {"the", "a"}
        result = []
        i = 0

        while i < len(entities):
            word = entities[i]
            if word.lower() in articles and i + 1 < len(entities):
                result.append(f"{word} {entities[i + 1]}")
                i += 2
            else:
                result.append(word)
                i += 1

        return result

    def capitalize(self, entities):
        lower_case_words = {"and", "the", "of", "in", "on", "at", "with", "a", "an", "for", "to", "by", "from"}

        def capitalize_title(title):
            words = title.split()
            capitalized_words = []
            for i, word in enumerate(words):
                if i == 0 or i == len(words) - 1 or word.lower() not in lower_case_words:
                    capitalized_words.append(word.capitalize())
                else:
                    capitalized_words.append(word.lower())
            return " ".join(capitalized_words)
        return [capitalize_title(title) for title in entities]


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