import datasets
import numpy as np
import pandas as pd
import torch.cuda
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flair.data import Sentence
from flair.models import SequenceTagger
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

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
    def dataset_builder(texts, labels, train_size=.8):
        train_texts, validation_texts, train_labels, validation_labels = train_test_split(texts, labels,
                                                                                            train_size=train_size,
                                                                                            stratify=labels)

        train_dict = {'Phrase': train_texts, 'Sentiment': train_labels}
        validation_dict = {'Phrase': validation_texts, 'Sentiment': validation_labels}

        train_dataset = datasets.Dataset.from_dict(train_dict)
        validation_dataset = datasets.Dataset.from_dict(validation_dict)

        dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                            "validation": validation_dataset})

        return dataset_dict

    def tokenize_fn(self, batch_item_dataset, apply_truncation=True, apply_padding=True):
        return self.tokenizer(batch_item_dataset["Phrase"], batch_item_dataset["Pos"], truncation=apply_truncation, padding=apply_padding)

    def pos_tagger_fn(self, batch_item_dataset):
        # make example sentence
        sentence = Sentence(batch_item_dataset["Phrase"])

        # predict NER tags
        self.tagger.predict(sentence)

        # print sentence
        tags = ' '.join(token.tag for token in sentence.tokens)

        return {'Pos': tags}
    
    def dataset_preprocessing(self, dataset_dict, apply_truncation=True, apply_padding=True):
        dataset_pos = dataset_dict.map(lambda single_item_dataset: self.pos_tagger_fn(single_item_dataset))
        dataset_tokenized = dataset_pos.map(lambda single_item_dataset: self.tokenize_fn(single_item_dataset, 
                                            apply_truncation=apply_truncation, apply_padding=apply_padding), batched=True)
        
        return dataset_tokenized
    
    def __prepare_trainer(self, train_texts, train_labels, batch_size: int = 16, num_train_epochs: int = 5):
        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            return {'sklearn_accuracy': accuracy_score(labels, predictions)}

        dataset = self.dataset_builder(train_texts, train_labels)
        dataset_preprocessed = self.dataset_preprocessing(dataset, apply_padding=False)

        # this specific model expects label column
        dataset_formatted = dataset_preprocessed.rename_column('Sentiment', 'label').remove_columns(['Phrase', "Pos"])

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments("../output/test-trainer",
                                        evaluation_strategy='epoch',
                                        num_train_epochs=num_train_epochs,
                                        optim='adamw_torch',
                                        per_device_train_batch_size=batch_size,
                                        per_device_eval_batch_size=batch_size,
                                        disable_tqdm=True,
                                        save_total_limit=3,
                                        save_strategy='epoch')

        trainer = Trainer(
            model_init=lambda: self.model_init(),
            args=training_args,
            train_dataset=dataset_formatted["train"],
            eval_dataset=dataset_formatted["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        return trainer

    def train(self, train_texts, train_labels, batch_size: int = 16, num_train_epochs: int = 5):

        trainer = self.__prepare_trainer(train_texts, train_labels, batch_size, num_train_epochs)
        trainer.train()

        return trainer
    
    def find_best_hyperparameters(self, trainer, cpu_number: int = 1, gpu_number: int = 1, n_trials: int = 1):

        tune_config = {
                "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                "num_train_epochs": tune.choice([2, 3, 4, 5]),
                "seed": tune.randint(0, 43),
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.uniform(1e-4, 5e-5),
                "lr_scheduler_type": tune.choice(['linear', 'cosine', 'polynomial'])
        }

        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="eval_sklearn_accuracy",
            mode="max",
            perturbation_interval=1,
            hyperparam_mutations=tune_config)

        best_trial = trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            direction="maximize",
            backend="ray",
            scheduler=pbt_scheduler,
            n_trials=n_trials,
            resources_per_trial={"cpu": cpu_number, "gpu": gpu_number},
            keep_checkpoints_num=1,
            local_dir="../hyper_search/",
            name="tune_transformer_pbt"
        )

        return best_trial

    def train_with_hyperparameters(self, texts, labels,
                                   batch_size: int = 16, num_train_epochs: int = 5,
                                   cpu_number: int = 1, gpu_number: int = 1, n_trials: int = 1):
        
        trainer = self.__prepare_trainer(texts, labels, batch_size, num_train_epochs)
        best_trial = self.find_best_hyperparameters(trainer, cpu_number, gpu_number, n_trials)

        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

        trainer.train()

        return trainer

    def compute_prediction(self, test_texts, test_ids, output_file='../submission_1.csv'):
        def compute_prediction(single_item):
            ids = torch.tensor(single_item['input_ids'], dtype=torch.int32).to(device).unsqueeze(dim=0)
            token_type_ids = torch.tensor(single_item['token_type_ids'], dtype=torch.int32).to(device).unsqueeze(dim=0)
            mask = torch.tensor(single_item['attention_mask'], dtype=torch.int32).to(device).unsqueeze(dim=0)

            logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=mask)
            prediction = np.argmax(torch.flatten(logits.logits).to('cpu').detach().numpy(), axis=-1)

            return {'pred': prediction}

        test_dict = {'Phrase': test_texts}
        dataset_dict = datasets.Dataset.from_dict(test_dict)
        dataset_preprocessed = self.dataset_preprocessing(dataset_dict)
        dataset_formatted = dataset_preprocessed.remove_columns(['Phrase', 'Pos'])

        result = dataset_formatted.map(compute_prediction)

        final_dict = {'PhraseId': test_ids, 'Sentiment': result['pred']}
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv(output_file, index=False)

        return result

# shortcut functions for quick training / testing

def train_experiment(approach: TransformersApproach, train_dataset_path: str, cut: int = None,
                    batch_size: int = 16, num_train_epochs: int = 5):
    
    train = pd.read_csv(train_dataset_path, sep="\t")
    train_texts = train['Phrase'].to_list()
    train_labels = train['Sentiment'].to_list()

    if cut is not None:
        train_texts = train_texts[:cut]
        train_labels = train_labels[:cut]

    return approach.train(train_texts, train_labels, batch_size, num_train_epochs)

def train_with_hyperparameters_tuning_experiment(approach: TransformersApproach, train_dataset_path: str, 
                                                cut: int = None,
                                                batch_size: int = 16, num_train_epochs: int = 5,
                                                cpu_number: int = 1, gpu_number: int = 1, n_trials: int = 1):
    
    train = pd.read_csv(train_dataset_path, sep="\t")
    train_texts = train['Phrase'].to_list()
    train_labels = train['Sentiment'].to_list()

    if cut is not None:
        train_texts = train_texts[:cut]
        train_labels = train_labels[:cut]

    return approach.train_with_hyperparameters(train_texts, train_labels,
                                                batch_size, num_train_epochs,
                                                cpu_number, gpu_number, n_trials)

def test_experiment(approach: TransformersApproach, test_dataset_path: str, output_file_path: str):

    test = pd.read_csv(test_dataset_path, sep="\t")

    test_texts = test['Phrase'].to_list()
    test_ids = test['PhraseId'].to_list()

    return approach.compute_prediction(test_texts, test_ids, output_file_path)


if __name__ == '__main__':

    train_path = '../dataset/train.tsv'
    test_path = '../dataset/test.tsv'
