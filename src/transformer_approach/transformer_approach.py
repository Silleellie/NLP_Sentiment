import numpy as np
import pandas as pd
import torch.cuda
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from src.transformer_approach.dataset_builder import CustomDataset

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

            return {'accuracy_metric': accuracy_metric.compute(predictions=predictions, references=labels)}

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(output_model_folder,
                                          evaluation_strategy='epoch',
                                          num_train_epochs=num_train_epochs,
                                          optim='adamw_torch',
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          disable_tqdm=True,
                                          save_total_limit=3,
                                          save_strategy='epoch')

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
            name="tune_transformer_pbt"
        )

        for n, v in best_trial.hyperparameters.items():
            setattr(trainer.args, n, v)

        # train on full dataset
        trainer.train_dataset = dataset_formatted['train']
        trainer.eval_dataset = dataset_formatted['validation']

        return trainer

    def compute_prediction(self, dataset_formatted, output_file='submission.csv'):
        def compute_single_prediction(single_item):
            ids = torch.tensor(single_item['input_ids'], dtype=torch.int32).to(device).unsqueeze(dim=0)
            token_type_ids = torch.tensor(single_item['token_type_ids'], dtype=torch.int32).to(device).unsqueeze(dim=0)
            mask = torch.tensor(single_item['attention_mask'], dtype=torch.int32).to(device).unsqueeze(dim=0)

            logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=mask)
            prediction = np.argmax(torch.flatten(logits.logits).to('cpu').detach().numpy(), axis=-1)

            return prediction

        # IMPORTANT!!! TO TEST THIS, WE APPLY DYNAMIC PADDING
        dataloader = DataLoader(dataset_formatted, collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer))
        final_pred = [compute_single_prediction(single_item) for single_item in dataloader]

        final_dict = {'PhraseId': list(dataset_formatted['PhraseId']), 'Sentiment': final_pred}
        final_df = pd.DataFrame(final_dict)
        final_df.to_csv(output_file, index=False)

        return final_dict


if __name__ == '__main__':
    train_path = '../../dataset/train.tsv'
    test_path = '../../dataset/test.tsv'

    t = TransformersApproach('bert-base-uncased')

    train_formatted = CustomDataset(train_path, cut=1000).preprocess(t.tokenizer, mode='only_phrase')

    t.train(train_formatted, batch_size=8, num_train_epochs=3, output_model_folder='output/test_model')

    # t.train_with_hyperparameters(train_formatted)

    test_formatted = CustomDataset(test_path).preprocess(t.tokenizer, mode='only_phrase')

    t.compute_prediction(test_formatted, output_file='submission.csv')
