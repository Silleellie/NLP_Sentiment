import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tqdm import tqdm

"""
Obtained accuracy results validation:

Custom Neural Network approach with default parameters: 0.6667948225041651
"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class CustomNetwork(nn.Module):
    def __init__(self, num_labels):
        super(CustomNetwork, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(0.5)

        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 192)
        self.linear3 = nn.Linear(192, num_labels)

    def forward(self, input):
        sequence_output = self.dropout(input)

        intermediate = self.linear1(sequence_output)
        intermediate = self.leaky_relu(intermediate.unsqueeze(dim=2))
        intermediate = self.dropout(intermediate)

        intermediate = self.linear2(intermediate.squeeze())
        intermediate = self.leaky_relu(intermediate.unsqueeze(dim=2))
        intermediate = self.dropout(intermediate)

        output = self.linear3(intermediate.squeeze())

        return output


class CustomEmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class SbertNetwork:

    def __init__(self, model_name='all-mpnet-base-v2', device="cuda:0"):
        self.net = CustomNetwork(5).to(device)
        self.model = SentenceTransformer(model_name, device=device)

    def train(self, train_texts, train_labels, batch_size=4,
              epochs=5, lr=5e-5, weight_d=1e-4, loss=nn.CrossEntropyLoss()):

        train_texts, validation_texts, train_labels, validation_labels = train_test_split(train_texts, train_labels,
                                                                                          train_size=.8,
                                                                                          stratify=train_labels,
                                                                                          random_state=42)

        train_embeddings = self.model.encode(train_texts, convert_to_tensor=True, device=device)
        validation_embeddings = self.model.encode(validation_texts, convert_to_tensor=True, device=device)

        criterion = loss
        optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_d)

        train_ds = CustomEmbeddingDataset(train_embeddings, train_labels)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        validation_ds = CustomEmbeddingDataset(validation_embeddings, validation_labels)
        validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=True)

        print('Starting Training')

        best_accuracy_score = 0

        for epoch in range(epochs):
            print("-------------------------------------------------")
            print("EPOCH: ", str(epoch + 1))
            running_loss = 0.0
            print("TRAIN")
            for (inputs, labels) in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print("train loss: ", str(running_loss / len(train_loader)))

            print("VALIDATION")
            accuracy = 0
            with torch.no_grad():
                for (inputs, labels) in tqdm(validation_loader):
                    inputs = inputs.to(device)

                    outputs = self.net(inputs)
                    predicted = torch.argmax(outputs.data, dim=1).cpu()
                    accuracy += accuracy_score(predicted, labels)

            accuracy = accuracy / len(validation_loader)
            print("accuracy score for current epoch: ", str(accuracy))
            print("previous best accuracy score: ", str(best_accuracy_score))
            if accuracy > best_accuracy_score:
                best_accuracy_score = accuracy
                torch.save(self.net.state_dict(), 'best_network_state.pth')

        print('Finished Training')
        print('FINAL BEST ACCURACY SCORE: ', str(best_accuracy_score))

    def test(self, test_texts):
        test_embeddings = self.model.encode(test_texts, convert_to_tensor=True, device=device)
        with torch.no_grad():
            outputs = self.net(test_embeddings)
            predictions = torch.argmax(outputs.data, dim=1).cpu()

        return predictions

    def load_model(self, path='best_network_state.pth'):
        self.net.load_state_dict(torch.load(path))

    def create_submission_csv(self, test_ids, predictions):
        final_dict = {'PhraseId': test_ids, 'Sentiment': predictions}
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv('submission.csv', index=False)

    def test_and_create_submission(self, test_texts, test_ids):
        self.create_submission_csv(test_ids, self.test(test_texts))


if __name__ == "__main__":
    train = pd.read_csv('../../dataset/train.tsv', sep="\t")
    test = pd.read_csv('../../dataset/test.tsv', sep="\t")
    train_texts = train['Phrase'].to_list()
    train_labels = train['Sentiment'].to_list()
    test_texts = test['Phrase'].to_list()
    test_ids = test['PhraseId'].to_list()

    model = SbertNetwork()
    model.train(train_texts, train_labels)
    model.load_model()
    model.test_and_create_submission(test_texts, test_ids)
