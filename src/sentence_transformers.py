from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.transformer_neural import CustomHead

model = SentenceTransformer('all-mpnet-base-v2')

train = pd.read_csv('../dataset/train.tsv', sep="\t")
texts = train['Phrase'].to_list()
labels = train['Sentiment'].to_list()

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels,
                                                                                train_size=.8,
                                                                                stratify=labels,
                                                                                random_state=42)

train_embeddings = model.encode(train_texts, convert_to_tensor=True)
test_embeddings = model.encode(test_texts, convert_to_tensor=True)

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


class CustomEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def custom_neural_network_approach(train_embeddings, train_labels, test_embeddings, batch_size = 4,
                                   epochs = 10, lr=5e-5, weight_d=1e-4, loss = nn.CrossEntropyLoss()):
    net = CustomHead(5).to("cuda:0")

    criterion = loss
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_d)

    ds = CustomEmbeddingDataset(train_embeddings, train_labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print('Starting Training')

    for epoch in range(epochs):
        print("EPOCH: ", str(epoch+1))
        running_loss = 0.0
        for (inputs, labels) in loader:

            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print('Finished Training')

    with torch.no_grad():
        outputs = net(test_embeddings)

    return torch.argmax(outputs.data, dim=1).cpu()

print("CLUSTERING APPROACH: ", accuracy_score(test_labels, clusters_approach(train_embeddings, train_labels, test_embeddings)))
print("MLP APPROAQCH: ", accuracy_score(test_labels, MLP_approach(train_embeddings, train_labels, test_embeddings)))
print("CUSTOM NEURAL NETWORK APPROACH: ", accuracy_score(test_labels, custom_neural_network_approach(train_embeddings, train_labels, test_embeddings)))

"""
Obtained accuracy results:

Clustering approach: 0.5264321414840446
MLP approach: 0.595924644367551
Custom Neural Network approach with default parameters: 0.6667948225041651
"""
