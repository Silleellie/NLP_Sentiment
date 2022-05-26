import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments

device = 'cuda:0'

class TransformersApproach:

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5,
                                                                        ignore_mismatched_sizes=True).to(device)

    @staticmethod
    def dataset_builder(raw_dataset_path):
        train = pd.read_csv(raw_dataset_path, sep="\t")
        all_texts = list(train['Phrase'])[:10000]
        all_labels = list(train['Sentiment'])[:10000]

        train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, train_size=.7)
        train_texts, validation_texts, train_labels, validation_labels = train_test_split(train_texts, train_labels,
                                                                                          train_size=.9)
        # 'label' needed by the model
        train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
        validation_dict = {'Phrase': validation_texts, 'Sentiment': validation_labels}
        test_dict = {'Phrase': test_texts, 'Sentiment': test_labels}

        train_dataset = datasets.Dataset.from_dict(train_dict)
        validation_dataset = datasets.Dataset.from_dict(validation_dict)
        test_dataset = datasets.Dataset.from_dict(test_dict)

        dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                             "validation": validation_dataset,
                                             "test": test_dataset})

        return dataset_dict

    def tokenize_fn(self, batch_item_dataset):
        return self.tokenizer(batch_item_dataset["Phrase"], truncation=True)


if __name__ == '__main__':
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return {'sklearn_accuracy': accuracy_score(labels, predictions)}

    t = TransformersApproach('bert-base-uncased')

    dataset = t.dataset_builder('../dataset/train.tsv')

    dataset_tokenized = dataset.map(lambda single_item_dataset: t.tokenize_fn(single_item_dataset),
                                    batched=True)

    # this specific model expects label column
    dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label').remove_columns('Phrase')

    data_collator = DataCollatorWithPadding(tokenizer=t.tokenizer)

    training_args = TrainingArguments("../output/test-trainer",
                                      evaluation_strategy='epoch',
                                      num_train_epochs=10,
                                      optim='adamw_torch',
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8)

    trainer = Trainer(
        t.model,
        training_args,
        train_dataset=dataset_formatted["train"],
        eval_dataset=dataset_formatted["validation"],
        data_collator=data_collator,
        tokenizer=t.tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
