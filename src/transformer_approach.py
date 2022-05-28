import datasets
import numpy as np
import pandas as pd
import torch.cuda
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

    def compute_prediction(self, output_file='./submission_1.csv'):
        def tokenize_fn_padding(tokenizer, batch_item_dataset):
            return tokenizer(batch_item_dataset["Phrase"], batch_item_dataset["Pos"], truncation=True, padding=True)

        def compute_prediction(single_item):
            ids = torch.tensor(single_item['input_ids'], dtype=torch.int32).to(device).unsqueeze(dim=0)
            token_type_ids = torch.tensor(single_item['token_type_ids'], dtype=torch.int32).to(device).unsqueeze(dim=0)
            mask = torch.tensor(single_item['attention_mask'], dtype=torch.int32).to(device).unsqueeze(dim=0)

            logits = self.model(input_ids=ids, token_type_ids=token_type_ids, attention_mask=mask)
            prediction = np.argmax(torch.flatten(logits.logits).to('cpu').detach().numpy(), axis=-1)

            return {'pred': prediction}

        test_set_path = '../dataset/test.tsv'

        test = pd.read_csv(test_set_path, sep="\t")

        test_dict = {'Phrase': list(test["Phrase"])}

        dataset_dict = datasets.Dataset.from_dict(test_dict)

        dataset_pos = dataset_dict.map(lambda single_item_dataset: self.pos_tagger_fn(single_item_dataset))

        dataset_tokenized = dataset_pos.map(lambda single_item_dataset: tokenize_fn_padding(self.tokenizer,
                                                                                            single_item_dataset),
                                            batched=True)

        # this specific model expects label column
        dataset_formatted = dataset_tokenized.remove_columns(['Phrase', "Pos"])

        result = dataset_formatted.map(compute_prediction)

        final_dict = {'PhraseId': test['PhraseId'], 'Sentiment': result['pred']}
        final_df = pd.DataFrame(final_dict)

        final_df.to_csv(output_file, index=False)


if __name__ == '__main__':

    t = TransformersApproach('checkpoint-15606')
    t.compute_prediction('submission_1.csv')
