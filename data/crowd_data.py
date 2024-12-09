import logging
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa


class CrowdData:
    def __init__(self):
        self.logger = logging.getLogger("crowd_data")
        self.file_path = "data/crowd_data.tsv"
        self.data = pd.read_csv(self.file_path, sep="\t")
        self.cleaned_data = self.clean_data()
        self.aggregated_answers = self.aggregate_answers()
        self.kappa_values = self.compute_fleiss_kappa()
        self.short_to_prefix = {"ddis:": "http://ddis.ch/atai/",
                                "wd:": "http://www.wikidata.org/entity/",
                                "wdt:": "http://www.wikidata.org/prop/direct/",
                                "schema:": "http://schema.org/"}

    def clean_data(self):
        self.data['LifetimeApprovalRate'] = self.data['LifetimeApprovalRate'].str.rstrip('%').astype(float) / 100
        cleaned_data = self.data[
            (self.data['LifetimeApprovalRate'] >= 0.5) &
            (self.data['WorkTimeInSeconds'] >= 5)
            ]

        return cleaned_data

    def aggregate_answers(self):
        aggregated = self.cleaned_data.groupby(
            ['HITId', 'Input1ID', 'Input2ID', 'Input3ID']
        ).agg({
            'AnswerLabel': lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN',
            'WorkerId': 'count'
        }).rename(columns={'WorkerId': 'Votes'})

        return aggregated.reset_index()

    def compute_fleiss_kappa(self):
        kappa_values = {}
        for batch_id, batch_data in self.cleaned_data.groupby('HITTypeId'):
            counts = pd.crosstab(batch_data['HITId'], batch_data['AnswerLabel'])
            kappa_values[batch_id] = fleiss_kappa(counts.values, method='fleiss')
        return kappa_values

    def get_answer_distribution(self, hit_id):
        specific_task = self.cleaned_data[self.cleaned_data['HITId'] == hit_id]
        distribution = specific_task['AnswerLabel'].value_counts().to_dict()
        return distribution

    def get_result(self, entity_uri, relation_uri):
        short_entity_uri = self.to_short_uri(entity_uri)
        short_relation_uri = self.to_short_uri(relation_uri)

        self.logger.debug(f"Searching for entity: '{short_entity_uri}', relation: '{short_relation_uri}'")

        hit_data = self.cleaned_data[
            (self.cleaned_data['Input1ID'] == short_entity_uri) &
            (self.cleaned_data['Input2ID'] == short_relation_uri)
            ]

        if hit_data.empty:
            self.logger.debug(f"No crowd data found.")
            return None

        def weighted_vote(group):
            correct_votes = group[group['AnswerLabel'] == 'CORRECT']
            incorrect_votes = group[group['AnswerLabel'] == 'INCORRECT']
            weighted_correct = correct_votes['LifetimeApprovalRate'].sum()
            weighted_incorrect = incorrect_votes['LifetimeApprovalRate'].sum()
            return weighted_correct - weighted_incorrect

        aggregated = hit_data.groupby('Input3ID').apply(weighted_vote)

        result = aggregated.idxmax()
        batch_id = hit_data.iloc[0]['HITTypeId']
        inter_rater = self.kappa_values[batch_id]
        hit_id = hit_data.iloc[0]['HITId']
        distribution = self.get_answer_distribution(hit_id)
        votes = ", ".join(
            [f"{count} {label.lower()} vote{'s' if count > 1 else ''}" for label, count in distribution.items()])

        result = self.to_full_uri(result)
        self.logger.debug(f"Found crowd data result {result}.")

        return result, inter_rater, votes

    def to_full_uri(self, item):
        item = str(item)
        for short, prefix in self.short_to_prefix.items():
            item = item.replace(short, prefix)
        return item

    def to_short_uri(self, item):
        item = str(item)
        for short, prefix in self.short_to_prefix.items():
            item = item.replace(prefix, short)
        return item


# Example usage
if __name__ == "__main__":
    processor = CrowdData()

    subject = "wd:Q11621"
    predicate = "wdt:P2142"
    obj = "792910554"
    response = processor.chatbot_response(subject, predicate, obj)
    print(response)
