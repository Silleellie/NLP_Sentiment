import itertools

import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig, DataCollatorWithPadding, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput
from tqdm.auto import tqdm
from datasets import load_metric
from transformers import get_scheduler

from src.utils.dataset_builder import CustomTrainValEvalHO, CustomTest

torch.cuda.empty_cache()

device = 'cuda:0'


class CustomHead(nn.Module):
    def __init__(self, num_labels):
        super(CustomHead, self).__init__()

        self.num_labels = num_labels

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.dropout = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(13, 13, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(13, 13, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(13)
        self.conv3 = nn.Conv2d(13, 13, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(13)
        self.conv4 = nn.Conv2d(13, 13, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(13)
        self.conv5 = nn.Conv2d(13, 13, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(13)
        self.linear1 = nn.Linear(13 * 1 * 24, num_labels)

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

        intermediate = self.conv5(intermediate)
        intermediate = self.leaky_relu(intermediate)
        intermediate = self.bn5(intermediate)

        intermediate = self.dropout(intermediate)
        intermediate = self.flatten(intermediate)

        output = self.linear1(intermediate)

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

        self.optim = AdamW(list(self.model.parameters()) + list(self.custom_head.parameters()),
                           lr=5e-5, weight_decay=1e-4)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):

        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # compute sentence embedding
        sentenced = torch.stack([torch.mean(tensor, dim=1) for tensor in outputs.hidden_states])
        sentenced = torch.permute(sentenced, (1, 0, 2)).unsqueeze(dim=2)

        logits = self.custom_head(sentenced)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)

    def trainer(self, n_epochs, train_dataloader, validation_dataloader, eval_dataloader, save_all=False):

        metric = load_metric("accuracy")

        num_training_steps = n_epochs * len(train_dataloader)

        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optim, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        best_eval_accuracy = 0

        for epoch in range(n_epochs):
            loss = 0
            mean_loss_acc = 0
            self.train()
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
            eval_accuracy = metric.compute()['accuracy']

            if not save_all:
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    torch.save(self, 'best_model.pth')
            else:
                torch.save(self, 'model_epoch_' + str(epoch) + ".pth")

            print({'eval_accuracy': eval_accuracy, 'loss_acc': mean_loss_acc, 'loss': loss.item()})

    def compute_prediction(self, dataset_formatted, output_file='submission.csv'):
        def compute_batch_prediction(batch_items):
            input_model_device = {k: v.to(device) for k, v in batch_items.items()}

            with torch.no_grad():
                logits = self(**input_model_device)

            prediction = torch.argmax(logits.logits, dim=-1).to('cpu')

            return [pred.item() for pred in prediction]

        phrase_ids = list(dataset_formatted['test']['PhraseId'])

        dataset_formatted_test_no_phrase_id = dataset_formatted['test'].remove_columns('PhraseId')

        dataloader = DataLoader(dataset_formatted_test_no_phrase_id,
                                collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
                                num_workers=2, batch_size=8)

        final_pred = list(itertools.chain.from_iterable([compute_batch_prediction(batch)
                                                         for batch in tqdm(dataloader)]))

        final_dict = {'PhraseId': phrase_ids, 'Sentiment': final_pred}
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv(output_file, index=False)

        return final_df


if __name__ == '__main__':

    train_path = '../../dataset/train.tsv'
    test_path = '../../dataset/test.tsv'

    cm = CustomModel('bert-base-uncased', num_labels=5)

    # ------------------- hold out splitting --------------------
    [dataset_dict] = CustomTrainValEvalHO(train_path).preprocess(cm.tokenizer, mode='with_pos')

    # ------- train with custom head for classification ---------
    data_collator = DataCollatorWithPadding(tokenizer=cm.tokenizer)

    train_dataloader = DataLoader(
        dataset_dict["train"], batch_size=8, collate_fn=data_collator, shuffle=True
    )

    validation_dataloader = DataLoader(
        dataset_dict["validation"], batch_size=8, collate_fn=data_collator, shuffle=True
    )

    eval_dataloader = DataLoader(
        dataset_dict["eval"], batch_size=8, collate_fn=data_collator
    )

    cm.trainer(n_epochs=3,
               train_dataloader=train_dataloader,
               validation_dataloader=validation_dataloader,
               eval_dataloader=eval_dataloader)

    # ----------------- build submission csv -------------------
    [formatted_dataset] = CustomTest(test_path).preprocess(cm.tokenizer, "with_pos")

    cm.compute_prediction(formatted_dataset, output_file='submission.csv')

