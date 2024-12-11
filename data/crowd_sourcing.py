import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa

class CrowdDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the processor with the file path to the TSV data.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep="\t")
        self.cleaned_data = None
        self.aggregated_answers = None
        self.kappa_values = None

    def clean_data(self):
        """
        Filters out malicious crowd workers based on AssignmentStatus and LifetimeApprovalRate.
        """
        self.data['LifetimeApprovalRate'] = self.data['LifetimeApprovalRate'].str.rstrip('%').astype(float) / 100
        self.cleaned_data = self.data[
            (self.data['LifetimeApprovalRate'] >= 0.5)
        ]
        return self.cleaned_data

    def aggregate_answers(self):
        """
        Aggregates answers using majority voting.
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned before aggregation.")

        aggregated = self.cleaned_data.groupby(
            ['HITId', 'Input1ID', 'Input2ID', 'Input3ID']
        ).agg({
            'AnswerLabel': lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNKNOWN',
            'WorkerId': 'count'
        }).rename(columns={'WorkerId': 'Votes'})

        self.aggregated_answers = aggregated.reset_index()
        return self.aggregated_answers

    def compute_fleiss_kappa(self):
        """
        Computes Fleiss' kappa for inter-rater agreement within each batch (HITTypeId).
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned before computing Fleiss' kappa.")

        self.kappa_values = {}
        for batch_id, batch_data in self.cleaned_data.groupby('HITTypeId'):
            counts = pd.crosstab(batch_data['HITId'], batch_data['AnswerLabel'])
            self.kappa_values[batch_id] = fleiss_kappa(counts.values, method='fleiss')
        return self.kappa_values

    def get_answer_distribution(self, hit_id):
        """
        Gets the distribution of answers for a specific HIT.
        """
        if self.cleaned_data is None:
            raise ValueError("Data must be cleaned before getting answer distribution.")

        specific_task = self.cleaned_data[self.cleaned_data['HITId'] == hit_id]
        distribution = specific_task['AnswerLabel'].value_counts().to_dict()
        return distribution

    def format_response(self, triple, majority_answer, inter_rater, distribution):
        """
        Formats the chatbot's response with the majority answer, inter-rater agreement, and answer distribution.
        """
        subject, predicate, obj = triple
        votes = ", ".join([f"{count} {label.lower()} vote{'s' if count > 1 else ''}" for label, count in distribution.items()])
        return (f"The {predicate} of {subject} is {obj}. "
                f"[Crowd, inter-rater agreement {inter_rater:.3f}, "
                f"The answer distribution for this specific task was {votes}]")

    def chatbot_response(self, subject, predicate, obj):
        """
        Generates a chatbot response for a given RDF triple.
        """
        if self.cleaned_data is None or self.aggregated_answers is None or self.kappa_values is None:
            raise ValueError("Data must be cleaned, aggregated, and inter-rater agreement computed before responding.")

        hit_data = self.cleaned_data[
            (self.cleaned_data['Input1ID'] == subject) &
            (self.cleaned_data['Input2ID'] == predicate) &
            (self.cleaned_data['Input3ID'] == obj)
        ]
        if not hit_data.empty:
            hit_id = hit_data.iloc[0]['HITId']
            majority_answer = self.aggregated_answers.loc[self.aggregated_answers['HITId'] == hit_id, 'AnswerLabel'].values[0]
            batch_id = hit_data.iloc[0]['HITTypeId']
            inter_rater = self.kappa_values[batch_id]
            distribution = self.get_answer_distribution(hit_id)
            return self.format_response((subject, predicate, obj), majority_answer, inter_rater, distribution)
        else:
            return f"No crowd data available for the {predicate} of {subject}."
