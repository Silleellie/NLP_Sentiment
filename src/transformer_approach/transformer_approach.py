import itertools

import shutil

import numpy as np
import pandas as pd
import torch.cuda
from datasets import load_metric
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from src.transformer_approach.dataset_builder import CustomTrainValHO, CustomTest, CustomTrainValKF
import wandb

wandb.init(project="Sentiment_analysis", entity="nlp_leshi")

# to disable wandb
# wandb.init(mode='disabled')

torch.cuda.empty_cache()

device = 'cuda:0'


class TransformersApproach:

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
        self.model = self.model_init().to(device)

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=5,
                                                                  ignore_mismatched_sizes=True)

    def _prepare_trainer(self, dataset_formatted, batch_size: int = 16, num_train_epochs: int = 5,
                         output_model_folder: str = 'output/test_trainer'):
        def compute_metrics(eval_preds, accuracy_metric):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)

            return accuracy_metric.compute(predictions=predictions, references=labels)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(output_model_folder,
                                          evaluation_strategy='epoch',
                                          num_train_epochs=num_train_epochs,
                                          optim='adamw_torch',
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          disable_tqdm=True,
                                          save_total_limit=3,
                                          save_strategy='epoch',
                                          report_to='wandb')

        accuracy_metric = load_metric("accuracy")

        trainer = Trainer(
            model_init=lambda: self.model_init(),
            args=training_args,
            train_dataset=dataset_formatted["train"],
            eval_dataset=dataset_formatted["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=lambda eval_preds: compute_metrics(eval_preds, accuracy_metric)
        )

        return trainer

    def train(self, dataset_formatted, batch_size: int = 16, num_train_epochs: int = 5,
              output_model_folder='output/test_trainer'):
        trainer = self._prepare_trainer(dataset_formatted, batch_size, num_train_epochs, output_model_folder)
        trainer.train()

        return trainer

    def train_with_hyperparameters(self, dataset_formatted,
                                   cpu_number: int = 1, gpu_number: int = 1, n_trials: int = 3,
                                   output_model_folder: str = 'output/test_trainer',
                                   output_hyper_folder: str = 'hyper_search'):
        # find hyperparameters on the 20 percent of the dataset
        len_train_20_percent = int(len(dataset_formatted['train']) * 0.2)
        len_validation_20_percent = int(len(dataset_formatted['validation']) * 0.2)
        dataset_shuffled = dataset_formatted.shuffle()

        cut_train = dataset_shuffled['train'].select(range(len_train_20_percent))
        cut_validation = dataset_shuffled['validation'].select(range(len_validation_20_percent))
        dataset_shuffled['train'] = cut_train
        dataset_shuffled['validation'] = cut_validation

        # parameters of the initialized trainer will be overridden with those of the best trial
        trainer = self._prepare_trainer(dataset_shuffled, output_model_folder=output_model_folder)

        # with hyperparmater you specify wandb in ray tune config
        # trainer.args.report_to = None

        tune_config = {
            # search space
            "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
            "num_train_epochs": tune.choice([2, 3, 4, 5]),
            "seed": tune.randint(0, 43),
            "weight_decay": tune.uniform(0.0, 0.3),
            "learning_rate": tune.uniform(1e-4, 5e-5),
            "lr_scheduler_type": tune.choice(['linear', 'cosine', 'polynomial']),

            # wandb configuration
            # "wandb": {
            #     "project": "Sentiment_analysis",
            #     "entity": "nlp_leshi",
            #     # "api_key": "b99fa531f482e6043fc5833d9e5ad81bb5d35c2f",
            #     "log_config": True
            # }
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

                # list and no 'choice()' otherwise continuous error
                "lr_scheduler_type": ['linear', 'cosine', 'polynomial']
            })

        best_trial = trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            direction="maximize",
            backend="ray",
            scheduler=pbt_scheduler,
            n_trials=n_trials,
            resources_per_trial={"cpu": cpu_number, "gpu": gpu_number},
            keep_checkpoints_num=1,
            local_dir=output_hyper_folder,
            name="tune_transformer_pbt",
            log_to_file=True,
            # loggers=DEFAULT_LOGGERS + (WandbLogger, )
        )

        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

        # train on full dataset
        trainer.train_dataset = dataset_formatted['train']
        trainer.eval_dataset = dataset_formatted['validation']

        # re enable wandb
        trainer.args.report_to = "wandb"

        return trainer

    def compute_prediction(self, dataset_formatted, output_file='submission.csv'):
        def compute_batch_prediction(single_item):
            ids = single_item['input_ids'].to(device)
            token_type_ids = single_item['token_type_ids'].to(device)
            mask = single_item['attention_mask'].to(device)

            with torch.no_grad():
                logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=mask)
            prediction = torch.argmax(logits.logits, dim=-1).to('cpu')

            return [pred.item() for pred in prediction]

        dataloader = DataLoader(dataset_formatted['test'], collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
                                num_workers=2, batch_size=8)
        final_pred = list(itertools.chain.from_iterable([compute_batch_prediction(batch)
                                                         for batch in tqdm(dataloader)]))

        final_dict = {'PhraseId': list(dataset_formatted['test']), 'Sentiment': final_pred}
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv(output_file, index=False)

        return final_df


if __name__ == '__main__':
    train_path = '../../dataset/train.tsv'
    test_path = '../../dataset/test.tsv'

    model_name = 'prajjwal1/bert-small'

    accuracy_metric = load_metric("accuracy")

    t = TransformersApproach(model_name)

    train_formatted_list = CustomTrainValKF(train_path, cut=1000, n_splits=2).preprocess(t.tokenizer,
                                                                                         mode='only_phrase')

    # for i, train_formatted in enumerate(train_formatted_list):
    #     model_name = model_name.replace('/', '_')
    #
    #     output_folder_split = f'output/{model_name}/test_split_{i}'
    #
    #     shutil.rmtree(output_folder_split, ignore_errors=True)
    #
    #     trainer = t.train(train_formatted, batch_size=2, num_train_epochs=2,
    #                       output_model_folder=output_folder_split)

    for i, train_formatted in enumerate(train_formatted_list):
        model_name = model_name.replace('/', '_')

        output_model_split = f'output/{model_name}/test_split_{i}'
        output_hyper_folder = f'output/{model_name}/test_split_{i}/hyper'

        shutil.rmtree(output_model_split, ignore_errors=True)

        trainer = t.train_with_hyperparameters(train_formatted, output_model_folder=output_model_split, n_trials=2,
                                               output_hyper_folder=output_hyper_folder)

    # test_formatted = CustomTest(test_path, cut=1000).preprocess(t.tokenizer, mode='only_phrase')
    #
    # t.compute_prediction(test_formatted, output_file='submission.csv')
