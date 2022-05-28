import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

from src.transformer_approach import TransformersApproach


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
            "lr_scheduler_type": ['linear', 'cosine', 'polynomial']  # list and no 'choice' otherwise continuous error
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


def my_final_train(model_name, train_file_path, best_trial):
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

    trainer.train()

    return trainer
