import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.transformers.transformers_neural import CustomHead

from tqdm import tqdm

torch.manual_seed(0)


class CustomEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def custom_neural_network_approach(train_embeddings, train_labels, validation_embeddings, validation_labels, 
                                   test_embeddings, batch_size = 4,
                                   epochs = 10, lr=5e-5, weight_d=1e-4, loss = nn.CrossEntropyLoss()):
    net = CustomHead(5).to("cuda:0")

    criterion = loss
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_d)

    train_ds = CustomEmbeddingDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    validation_ds = CustomEmbeddingDataset(validation_embeddings, validation_labels)
    validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)

    print('Starting Training')

    best_accuracy_score = 0

    for epoch in range(epochs):
        print("-------------------------------------------------")
        print("EPOCH: ", str(epoch+1))
        running_loss = 0.0
        print("TRAIN")
        for (inputs, labels) in tqdm(train_loader):

            inputs = inputs.to("cuda:0")
            labels = labels.to("cuda:0")

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print("train loss: ", str(running_loss / len(train_loader)))

        print("VALIDATION")
        accuracy = 0
        with torch.no_grad():
            for (inputs, labels) in tqdm(validation_loader):

                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")

                outputs = net(validation_embeddings)
                predicted = torch.argmax(outputs.data, dim=1).cpu()
                accuracy += accuracy_score(predicted, validation_labels)

        accuracy = accuracy / len(validation_loader)
        print("accuracy score for current epoch: ", str(accuracy))
        print("previous best accuracy score: ", str(best_accuracy_score))
        if accuracy > best_accuracy_score:
            best_accuracy_score = accuracy
            torch.save(net.state_dict(), 'network_state.pth')

    print('Finished Training')
    print('FINAL BEST ACCURACY SCORE: ', str(best_accuracy_score))

    net = CustomHead(5).to("cuda:0")
    net.load_state_dict(torch.load('network_state.pth'))
    with torch.no_grad():
        outputs = net(test_embeddings)
        predictions = torch.argmax(outputs.data, dim=1).cpu()
    
    return predictions


if __name__ == "__main__":

    train = pd.read_csv('../dataset/train.tsv', sep="\t")
    test = pd.read_csv('../dataset/test.tsv', sep="\t")
    train_texts = train['Phrase'].to_list()
    train_labels = train['Sentiment'].to_list()
    test_texts = test['Phrase'].to_list()
    test_ids = test['PhraseId'].to_list()

    train_texts, validation_texts, train_labels, validation_labels = train_test_split(train_texts, train_labels,
                                                                                      train_size=.8,
                                                                                      stratify=train_labels,
                                                                                      random_state=42)
    
    model = SentenceTransformer('all-mpnet-base-v2')
    train_embeddings = model.encode(train_texts, convert_to_tensor=True)
    validation_embeddings = model.encode(validation_texts, convert_to_tensor=True)
    test_embeddings = model.encode(test_texts, convert_to_tensor=True)

    predictions = custom_neural_network_approach(train_embeddings, train_labels,
                                                 validation_embeddings, validation_labels,
                                                 test_embeddings)
    
    final_dict = {'PhraseId': test_ids, 'Sentiment': predictions}
    final_df = pd.DataFrame(final_dict)
    final_df.to_csv('submission.csv', index=False)