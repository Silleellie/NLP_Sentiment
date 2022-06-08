from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Obtained accuracy results:

Clustering approach: 0.5264321414840446
MLP approach: 0.595924644367551
"""

def clusters_approach(train_embeddings, train_labels, test_embeddings):
    clusters = {}
    for label in [0, 1, 2, 3, 4]:
        clusters[label] = []

    for train_label, train_embedding in zip(train_labels, train_embeddings.cpu()):
        clusters[train_label].append(train_embedding)

    for label in clusters.keys():
        clusters[label] = torch.mean(torch.stack(clusters[label]), axis=0)

    predictions = []
    for test_embedding in test_embeddings.cpu():
        scores = {}
        for train_label in clusters.keys():
            scores[train_label] = util.cos_sim(test_embedding, clusters[train_label])
        predictions.append(max(scores, key=scores.get))
    
    return predictions

def MLP_approach(train_embeddings, train_labels, test_embeddings):
    clf = MLPClassifier(random_state=42, max_iter=500).fit(train_embeddings.cpu(), train_labels)
    return clf.predict(test_embeddings.cpu())


if __name__ == "__main__":

    """
    These approaches were not considered for the real test set since the accuracy
    obtained on the experiment was too low
    """
        
    model = SentenceTransformer('all-mpnet-base-v2')

    train = pd.read_csv('../../dataset/train.tsv', sep="\t")
    texts = train['Phrase'].to_list()[:10000]
    labels = train['Sentiment'].to_list()[:10000]

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels,
                                                                            train_size=.8,
                                                                            stratify=labels,
                                                                            random_state=42)

    train_embeddings = model.encode(train_texts, convert_to_tensor=True)
    test_embeddings = model.encode(test_texts, convert_to_tensor=True)

    print("CLUSTERING APPROACH: ", accuracy_score(test_labels, clusters_approach(train_embeddings, train_labels, test_embeddings)))
    print("MLP APPROACH: ", accuracy_score(test_labels, MLP_approach(train_embeddings, train_labels, test_embeddings)))
