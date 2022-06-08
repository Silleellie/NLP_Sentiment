from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod

"""
Obtained accuracy results validation:

Clustering approach: 0.5264321414840446
MLP approach: 0.595924644367551
"""

class Approach(ABC):

    def __init__(self, model_name = 'all-mpnet-base-v2', device = 'cpu') -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)
    
    @abstractmethod
    def fit(self, train_embeddings, train_labels):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, test_embeddings):
        raise NotImplementedError

class ClusteringApproach(Approach):

    def __init__(self, model_name = 'all-mpnet-base-v2', device = 'cpu') -> None:
        super().__init__(model_name, device)
        self.clusters = {}

    def fit(self, train_texts, train_labels):
        train_embeddings = self.model.encode(train_texts, convert_to_tensor=True)

        for label in np.unique(train_labels):
            self.clusters[label] = []

        for train_label, train_embedding in zip(train_labels, train_embeddings):
            self.clusters[train_label].append(train_embedding)

        for label in self.clusters.keys():
            self.clusters[label] = torch.mean(torch.stack(self.clusters[label]), axis=0)
    
    def predict(self, test_texts):
        test_embeddings = self.model.encode(test_texts, convert_to_tensor=True)

        predictions = []
        for test_embedding in test_embeddings:
            scores = {}
            for train_label in self.clusters.keys():
                scores[train_label] = util.cos_sim(test_embedding, self.clusters[train_label])
            predictions.append(max(scores, key=scores.get))
        
        return predictions

class MLPApproach(Approach):

    def __init__(self, model_name = 'all-mpnet-base-v2', device = 'cpu', **args) -> None:
        super().__init__(model_name, device)
        self.clf = MLPClassifier(random_state=42, max_iter=500, **args)
    
    def fit(self, train_texts, train_labels):
        train_embeddings = self.model.encode(train_texts, convert_to_tensor=True)
        self.clf.fit(train_embeddings, train_labels)
    
    def predict(self, test_texts):
        test_embeddings = self.model.encode(test_texts, convert_to_tensor=True)
        return self.clf.predict(test_embeddings)


if __name__ == "__main__":

    """
    These approaches were not considered for the real test set since the accuracy
    obtained on the experiment was too low
    """

    train = pd.read_csv('../../dataset/train.tsv', sep="\t")
    texts = train['Phrase'].to_list()[:10000]
    labels = train['Sentiment'].to_list()[:10000]

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels,
                                                                            train_size=.8,
                                                                            stratify=labels,
                                                                            random_state=42)

    mlp = MLPApproach()
    clustering = ClusteringApproach()

    mlp.fit(train_texts, train_labels)
    mlp_predictions = mlp.predict(test_texts)

    clustering.fit(train_texts, train_labels)
    clustering_predictions = clustering.predict(test_texts)

    print("CLUSTERING APPROACH: ", accuracy_score(test_labels, clustering_predictions))
    print("MLP APPROACH: ", accuracy_score(test_labels, mlp_predictions))
