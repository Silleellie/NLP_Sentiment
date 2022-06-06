from copy import deepcopy

import datasets
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, DataCollatorWithPadding, AutoTokenizer, get_scheduler, \
    AutoModelForSequenceClassification
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from tqdm.auto import tqdm
from datasets import load_metric
from transformers import get_scheduler

torch.cuda.empty_cache()

device = 'cuda:0'


class CustomHead(nn.Module):
    def __init__(self, num_labels):
        super(CustomHead, self).__init__()

        self.num_labels = num_labels

        self.leaky_relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.linear1 = nn.Conv2d(13, 26, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(26)
        self.linear2 = nn.Conv2d(26, 42, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(42)
        self.linear3 = nn.Conv2d(42, 84, kernel_size=3, stride=2, padding=1)
        self.linear4 = nn.Linear(84 * 1 * 96, num_labels)

    def forward(self, input):

        intermediate = self.linear1(input)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn1(intermediate)

        intermediate = self.linear2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn2(intermediate)

        intermediate = self.linear3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.dropout(intermediate)

        intermediate = self.flatten(intermediate)

        output = self.linear4(intermediate)

        return output


class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CustomModel, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint,
                                                                                             output_attentions=True,
                                                                                             output_hidden_states=True))
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.custom_head = CustomHead(num_labels).to(device)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sentenced = torch.stack([torch.mean(tensor, dim=1) for tensor in outputs.hidden_states])
        sentenced = torch.permute(sentenced, (1, 0, 2)).unsqueeze(dim=2)

        # sentenced = torch.stack(outputs.hidden_states)
        # sentenced = torch.permute(sentenced, (1, 0, 2, 3))

        logits = self.custom_head(sentenced)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


class Trainer:

    def __init__(self):
        self.model = CustomModel('bert-base-uncased', 5)

        self.optim = AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)

    def trainer(self, n_epochs, train_dataloader, validation_dataloader):

        metric = load_metric("accuracy")

        num_training_steps = n_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optim, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        for epoch in range(n_epochs):
            loss = 0
            self.model.train()
            for batch in tqdm(train_dataloader):
                self.optim.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                predictions = torch.argmax(outputs.logits, dim=-1)
                loss_acc = -metric.compute(predictions=predictions, references=batch['labels'])['accuracy']
                loss_acc = torch.tensor(loss_acc, requires_grad=True).to(device)

                loss_acc.backward()

                self.optim.step()
                lr_scheduler.step()

            self.model.eval()
            for batch in tqdm(validation_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch['labels'])

            print({**metric.compute(), **{'loss_acc': loss_acc, 'loss': loss}})


def tokenize_fn(tokenizer, batch_item_dataset):
    return tokenizer(batch_item_dataset["Phrase"])


if __name__ == '__main__':
    train = pd.read_csv('../dataset/train.tsv', sep="\t")
    texts = train['Phrase'].to_list()[:5000]
    labels = train['Sentiment'].to_list()[:5000]

    train_texts, validation_texts, train_labels, validation_labels = train_test_split(texts, labels,
                                                                                      train_size=0.8,
                                                                                      stratify=labels)

    train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
    validation_dict = {'Phrase': validation_texts, 'Sentiment': validation_labels}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    validation_dataset = datasets.Dataset.from_dict(validation_dict)

    dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                         "validation": validation_dataset})

    cm = Trainer()

    tokenized_dataset = dataset_dict.map(lambda batch: tokenize_fn(cm.model.tokenizer, batch), batched=True)

    formatted_dataset = tokenized_dataset.rename_column('Sentiment', 'label').remove_columns('Phrase')

    # from list to tensors
    formatted_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=cm.model.tokenizer)

    train_dataloader = DataLoader(
        formatted_dataset["train"], batch_size=8, collate_fn=data_collator
    )

    validation_dataloader = DataLoader(
        formatted_dataset["validation"], batch_size=8, collate_fn=data_collator
    )

    cm.trainer(3, train_dataloader, validation_dataloader)
