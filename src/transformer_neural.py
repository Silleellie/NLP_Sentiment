import os.path
import random
from copy import deepcopy

import datasets
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ExponentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
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

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(13, 26, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(26, 42, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(42)
        self.conv3 = nn.Conv2d(42, 84, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(84)
        self.conv4 = nn.Conv2d(84, 168, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(168)
        self.linear1 = nn.Linear(168, 84)
        self.linear2 = nn.Linear(84, num_labels)

    def forward(self, input):
        intermediate = self.conv1(input)
        intermediate = self.leaky_relu(intermediate)

        intermediate = self.conv2(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn2(intermediate)

        intermediate = self.conv3(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn3(intermediate)

        intermediate = self.conv4(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn4(intermediate)

        intermediate = self.linear1(intermediate)
        intermediate = self.dropout(intermediate)

        intermediate = self.flatten(intermediate)

        output = self.linear2(intermediate)

        return output


class CustomModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(CustomModel, self).__init__()

        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint,
                                                                                             output_hidden_states=True))
        self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.custom_head = CustomHead(num_labels).to(device)

        self.optim = AdamW(self.model.parameters(), lr=5e-5, weight_decay=1e-4)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sentenced = torch.stack([torch.mean(tensor, dim=1) for tensor in outputs.hidden_states])
        sentenced = torch.permute(sentenced, (1, 0, 2)).unsqueeze(dim=2)

        logits = self.custom_head(sentenced)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

    def trainer(self, n_epochs, train_dataloader, validation_dataloader, eval_dataloader):

        metric = load_metric("accuracy")

        num_training_steps = n_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optim, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        best_eval_accuracy = 0

        for epoch in range(n_epochs):
            loss = 0
            loss_acc = 0
            self.train()
            mean_loss_acc = 0
            for batch in tqdm(train_dataloader):
                self.optim.zero_grad()

                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self(**batch)
                loss = outputs.loss
                loss.backward()

                for batch_val in validation_dataloader:

                    # since random shuffle this will be always different
                    batch_val = {k: v.to(device) for k, v in batch_val.items()}

                    with torch.no_grad():
                        output_val = self(**batch_val)

                    predictions = torch.argmax(output_val.logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch_val['labels'])

                    # only one batch
                    break

                loss_acc = -metric.compute()['accuracy']

                mean_loss_acc += loss_acc

                loss_acc = torch.tensor(loss_acc, requires_grad=True).to(device)

                loss_acc.backward()

                self.optim.step()
                lr_scheduler.step()

            self.eval()
            for batch in tqdm(eval_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch['labels'])
            

            mean_loss_acc = mean_loss_acc / len(train_dataloader)
            eval_accuracy = metric.compute()

            if eval_accuracy > best_eval_accuracy:
                torch.save(self, 'best_model.pth')
            print({**eval_accuracy, **{'loss_acc': mean_loss_acc, 'loss': loss.item()}})


def tokenize_fn(tokenizer, batch_item_dataset):
    return tokenizer(batch_item_dataset["Phrase"])


if __name__ == '__main__':
    train = pd.read_csv('../dataset/train.tsv', sep="\t")
    texts = train['Phrase'].to_list()[:10000]
    labels = train['Sentiment'].to_list()[:10000]

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(texts, labels,
                                                                          train_size=0.8,
                                                                          stratify=labels,
                                                                          shuffle=True)

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels,
                                                                        train_size=0.9,
                                                                        stratify=train_labels,
                                                                        shuffle=True)

    train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
    validation_dict = {'Phrase': val_texts, 'Sentiment': val_labels}
    eval_dict = {'Phrase': eval_texts, 'Sentiment': eval_labels}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    validation_dataset = datasets.Dataset.from_dict(validation_dict)
    eval_dataset = datasets.Dataset.from_dict(eval_dict)

    dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                         "validation": validation_dataset,
                                         "eval": eval_dataset})

    cm = CustomModel('bert-base-uncased', 5)

    tokenized_dataset = dataset_dict.map(lambda batch: tokenize_fn(cm.tokenizer, batch), batched=True)

    formatted_dataset = tokenized_dataset.rename_column('Sentiment', 'label').remove_columns('Phrase')

    # from list to tensors
    formatted_dataset.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=cm.tokenizer)

    # class_sample_count = np.array(
    #     [len(np.where(formatted_dataset['train']['label'] == t)[0]) for t in np.unique(formatted_dataset['train']['label'])])
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[t] for t in formatted_dataset['train']['label']])
    #
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_dataloader = DataLoader(
        formatted_dataset["train"], batch_size=16, collate_fn=data_collator, shuffle=True
    )

    validation_dataloader = DataLoader(
        formatted_dataset["validation"], batch_size=16, collate_fn=data_collator, shuffle=True
    )

    eval_dataloader = DataLoader(
        formatted_dataset["eval"], batch_size=16, collate_fn=data_collator
    )

    cm.trainer(1, train_dataloader, validation_dataloader, eval_dataloader)

    print("we")
