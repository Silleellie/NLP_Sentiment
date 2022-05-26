import os
from typing import Optional, List

import numpy as np
from ray import tune

import transformers
from sklearn.metrics import accuracy_score
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

import torch

from src.transformer_approach import TransformersApproach


class TuneTransformerTrainer(transformers.Trainer):
    def get_optimizers(self):
        self.current_optimizer = self.optimizer
        self.current_scheduler = self.lr_scheduler
        return (self.current_optimizer, self.current_scheduler)

    def evaluate(self, eval_dataset=None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval"):

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(eval_dataloader,
                                      description="Evaluation",
                                      ignore_keys=ignore_keys,
                                      metric_key_prefix=metric_key_prefix)
        self.log(output.metrics)

        self.save_state()

        tune.report(**output.metrics)

        return output.metrics

    def save_state(self):
        with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
            self.args.output_dir = checkpoint_dir
            # This is the directory name that Huggingface requires.
            output_dir = os.path.join(
                self.args.output_dir,
                f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.save_model(output_dir)
            if self.is_world_process_zero():
                torch.save(self.current_optimizer.state_dict(),
                           os.path.join(output_dir, "optimizer.pt"))
                torch.save(self.current_scheduler.state_dict(),
                           os.path.join(output_dir, "scheduler.pt"))


def recover_checkpoint(tune_checkpoint_dir, model_name=None):
    if tune_checkpoint_dir is None or len(tune_checkpoint_dir) == 0:
        return model_name
    # Get subdirectory used for Huggingface.
    subdirs = [
        os.path.join(tune_checkpoint_dir, name)
        for name in os.listdir(tune_checkpoint_dir)
        if os.path.isdir(os.path.join(tune_checkpoint_dir, name))
    ]
    # There should only be 1 subdir.
    assert len(subdirs) == 1, subdirs
    return subdirs[0]


def train_transformer_hyperparam(config, checkpoint_dir=None):
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        return {'sklearn_accuracy': accuracy_score(labels, predictions)}

    training_args = TrainingArguments(
        output_dir=tune.get_trial_dir(),
        learning_rate=config["learning_rate"],
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        # We explicitly set save to 0, and do checkpointing in evaluate instead
        save_steps=0,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_gpu_train_batch_size"],
        per_device_eval_batch_size=config["per_gpu_val_batch_size"],
        warmup_steps=0,
        weight_decay=config["weight_decay"],
        logging_dir="./logs",
    )

    model_name_or_path = recover_checkpoint(checkpoint_dir, config["model_name"])

    t = TransformersApproach(model_name_or_path)

    dataset = config['dataset']

    dataset_pos = dataset.map(lambda single_item_dataset: t.pos_tagger_fn(single_item_dataset))

    dataset_tokenized = dataset_pos.map(lambda single_item_dataset: t.tokenize_fn(single_item_dataset),
                                        batched=True)

    # this specific model expects label column
    dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label').remove_columns(['Phrase', 'Pos'])

    data_collator = DataCollatorWithPadding(tokenizer=t.tokenizer)

    # Use our modified TuneTransformerTrainer
    tune_trainer = TuneTransformerTrainer(
        model=t.model,
        args=training_args,
        train_dataset=dataset_formatted['train'],
        eval_dataset=dataset_formatted['validation'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    tune_trainer.train()


def hyper_run(model_name):

    dataset = TransformersApproach.dataset_builder('../dataset/train.tsv', cut=100)

    config = {
        # These 3 configs below were defined earlier
        "model_name": model_name,
        "data_dir": "test",
        "per_gpu_val_batch_size": 32,
        "per_gpu_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "weight_decay": tune.uniform(0.0, 0.3),
        "num_epochs": tune.choice([2, 3, 4, 5]),
        "dataset": dataset
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_sklearn_accuracy",
        mode="max",
        perturbation_interval=2,
        hyperparam_mutations={
            "weight_decay": lambda: tune.uniform(0.0, 0.3).func(None),
            "learning_rate": lambda: tune.uniform(1e-5, 5e-5).func(None),
            "per_gpu_train_batch_size": [16, 32, 64],
        })

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "per_gpu_train_batch_size": "train_bs/gpu",
            "num_epochs": "num_epochs"
        },
        metric_columns=[
            "eval_sklearn_accuracy", "eval_loss", "epoch", "training_iteration"
        ])

    analysis = tune.run(
        train_transformer_hyperparam,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        config=config,
        num_samples=10,
        scheduler=scheduler,
        keep_checkpoints_num=3,
        checkpoint_score_attr="training_iteration",
        progress_reporter=reporter,
        local_dir="./ray_results/",
        name="tune_transformer_pbt")

    best_config = analysis.get_best_config(metric="eval_sklearn_accuracy", mode="max")
    print(best_config)


if __name__ == '__main__':
    hyper_run('albert-base-v2')
