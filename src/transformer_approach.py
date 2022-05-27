import datasets
import numpy as np
import pandas as pd
import torch.cuda
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from flair.data import Sentence
from flair.models import SequenceTagger

torch.cuda.empty_cache()

device = 'cuda:0'


class TransformersApproach:

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.model = self.model_init()
        # load tagger
        self.tagger = SequenceTagger.load("flair/pos-english-fast")

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=5,
                                                                  ignore_mismatched_sizes=True).to(device)

    @staticmethod
    def dataset_builder(raw_dataset_path, cut=None, with_test_set=False):
        train = pd.read_csv(raw_dataset_path, sep="\t")
        all_texts = list(train['Phrase'])[:cut]
        all_labels = list(train['Sentiment'])[:cut]

        if with_test_set:
            train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, train_size=.7,
                                                                                  stratify=all_labels)

            train_texts, validation_texts, train_labels, validation_labels = train_test_split(train_texts, train_labels,
                                                                                              train_size=.8,
                                                                                              stratify=train_labels)

            train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
            test_dict = {'Phrase': test_texts, 'Sentiment': test_labels}
            validation_dict = {'Phrase': validation_texts, 'Sentiment': validation_labels}

            train_dataset = datasets.Dataset.from_dict(train_dict)
            test_dataset = datasets.Dataset.from_dict(test_dict)
            validation_dataset = datasets.Dataset.from_dict(validation_dict)

            dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                                 "validation": validation_dataset,
                                                 "test": test_dataset})
        else:
            train_texts, validation_texts, train_labels, validation_labels = train_test_split(all_texts, all_labels,
                                                                                              train_size=.8,
                                                                                              stratify=all_labels)

            train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
            validation_dict = {'Phrase': validation_texts, 'Sentiment': validation_labels}

            train_dataset = datasets.Dataset.from_dict(train_dict)
            validation_dataset = datasets.Dataset.from_dict(validation_dict)

            dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                                 "validation": validation_dataset})

        return dataset_dict

    def tokenize_fn(self, batch_item_dataset):
        return self.tokenizer(batch_item_dataset["Phrase"], batch_item_dataset["Pos"], truncation=True)

    def pos_tagger_fn(self, batch_item_dataset):
        # make example sentence
        sentence = Sentence(batch_item_dataset["Phrase"])

        # predict NER tags
        self.tagger.predict(sentence)

        # print sentence
        tags = ' '.join(token.tag for token in sentence.tokens)

        return {'Pos': tags}


def run_hyperparameters(model_name, train_file_path, cpu_number, gpu_number):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return {'sklearn_accuracy': accuracy_score(labels, predictions)}

    t = TransformersApproach(model_name)

    dataset = t.dataset_builder(train_file_path, cut=15000)

    dataset_pos = dataset.map(lambda single_item_dataset: t.pos_tagger_fn(single_item_dataset))

    dataset_tokenized = dataset_pos.map(lambda single_item_dataset: t.tokenize_fn(single_item_dataset),
                                        batched=True)

    # this specific model expects label column
    dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label').remove_columns(['Phrase', "Pos"])

    data_collator = DataCollatorWithPadding(tokenizer=t.tokenizer)

    training_args = TrainingArguments("../output/test-trainer",
                                      evaluation_strategy='epoch',
                                      num_train_epochs=5,
                                      optim='adamw_torch',
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      disable_tqdm=True,
                                      save_total_limit=3,
                                      save_strategy='epoch')

    trainer = Trainer(
        model_init=lambda: t.model_init(),
        args=training_args,
        train_dataset=dataset_formatted["train"],
        eval_dataset=dataset_formatted["validation"],
        data_collator=data_collator,
        tokenizer=t.tokenizer,
        compute_metrics=compute_metrics
    )

    tune_config = {
        "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "seed": tune.randint(0, 43),
        "weight_decay": tune.uniform(0.0, 0.3),
        "learning_rate": tune.uniform(1e-4, 5e-5),
        "lr_scheduler_type": ['linear', 'cosine', 'polynomial']
    }

    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_sklearn_accuracy",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
            "num_train_epochs": [2, 3, 4, 5],
            "seed": tune.randint(0, 43),
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-4, 5e-5),
            "lr_scheduler_type": ['linear', 'cosine', 'polynomial']
        })

    best_trial = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        direction="maximize",
        backend="ray",
        scheduler=pbt_scheduler,
        n_trials=6,
        resources_per_trial={"cpu": cpu_number, "gpu": gpu_number},
        keep_checkpoints_num=1,
        local_dir="../hyper_search/",
        name="tune_transformer_pbt"
    )

    return best_trial


def final_train(model_name, train_file_path, best_trial):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return {'sklearn_accuracy': accuracy_score(labels, predictions)}

    t = TransformersApproach(model_name)

    dataset = t.dataset_builder(train_file_path)

    dataset_pos = dataset.map(lambda single_item_dataset: t.pos_tagger_fn(single_item_dataset))

    dataset_tokenized = dataset_pos.map(lambda single_item_dataset: t.tokenize_fn(single_item_dataset),
                                        batched=True)

    # this specific model expects label column
    dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label').remove_columns(['Phrase', "Pos"])

    data_collator = DataCollatorWithPadding(tokenizer=t.tokenizer)

    training_args = TrainingArguments("../output/test-trainer",
                                      evaluation_strategy='epoch',
                                      num_train_epochs=5,
                                      optim='adamw_torch',
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      disable_tqdm=True,
                                      save_total_limit=3,
                                      save_strategy='epoch')

    trainer = Trainer(
        model_init=lambda: t.model_init(),
        args=training_args,
        train_dataset=dataset_formatted["train"],
        eval_dataset=dataset_formatted["validation"],
        data_collator=data_collator,
        tokenizer=t.tokenizer,
        compute_metrics=compute_metrics
    )

    for n, v in best_trial.hyperparameters.items():
        setattr(trainer.args, n, v)

    return trainer.train()


if __name__ == '__main__':
    run_hyperparameters('bert-base-uncased', '../dataset/train.tsv', 4, 1)
