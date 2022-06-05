from abc import abstractmethod
from typing import List

import datasets
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.model_selection import train_test_split, StratifiedKFold

# load tagger as class attribute
tagger = SequenceTagger.load("flair/pos-english-fast")


class CustomDataset:
    def __init__(self, tsv_path: str, cut: int = None):
        df = pd.read_csv(tsv_path, sep='\t')

        self.dataset_dict_list = self._build_dataset(df, cut)

    @abstractmethod
    def _build_dataset(self, df: pd.DataFrame, cut: int) -> List[datasets.DatasetDict]:
        raise NotImplementedError

    def preprocess(self, tokenizer, mode='only_phrase') -> List[datasets.DatasetDict]:
        dataset_tokenized = None

        dataset_formatted_list = []
        for dataset_dict in self.dataset_dict_list:
            if mode == 'only_phrase':

                dataset_tokenized = dataset_dict.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item),
                                                     batched=True)
            elif mode == 'with_pos':

                dataset_pos = dataset_dict.map(lambda batch_item: self.pos_tagger_fn(batch_item))
                dataset_tokenized = dataset_pos.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item,
                                                                                         field_2="Pos"),
                                                    batched=True)
                dataset_tokenized = dataset_tokenized.remove_columns('Pos')
            elif mode == 'with_reference':
                dataset_tokenized = dataset_dict.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item,
                                                                                          field_2="OriginalSentence"),
                                                     batched=True)

            dataset_formatted = self._format_dataset(dataset_tokenized)
            dataset_formatted_list.append(dataset_formatted)

        return dataset_formatted_list

    @abstractmethod
    def _format_dataset(self, dataset_tokenized: datasets.DatasetDict) -> datasets.DatasetDict:
        raise NotImplementedError

    @staticmethod
    def tokenizer_fn(tokenizer, batch_item, field_2=None):
        if field_2 is not None:
            return tokenizer(batch_item["Phrase"], batch_item[field_2], truncation=True)

        return tokenizer(batch_item['Phrase'])

    @staticmethod
    def pos_tagger_fn(single_item):
        # make example sentence
        sentence = Sentence(single_item["Phrase"])

        # predict NER tags
        tagger.predict(sentence)

        # print sentence
        tags = ' '.join(token.tag for token in sentence.tokens)

        return {'Pos': tags}


class CustomTrainValKF(CustomDataset):

    def __init__(self, tsv_path: str, cut: int = None, n_splits: int = 5):
        self.n_splits = n_splits

        super().__init__(tsv_path, cut)


    def _build_dataset(self, df: pd.DataFrame, cut: int) -> List[datasets.DatasetDict]:

        all_original_sentences = df.groupby(by='SentenceId', as_index=False).first()
        all_original_sentences.drop(columns=['PhraseId', 'Sentiment'], inplace=True)
        all_original_sentences.rename(columns={'Phrase': 'OriginalSentence'}, inplace=True)

        expanded_df = df.merge(all_original_sentences, on='SentenceId')[:cut]

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        split_list = list(kf.split(X=expanded_df, y=expanded_df['Sentiment']))

        datasets_list = []
        for split in split_list:
            train_df = expanded_df.iloc[split[0]]
            validation_df = expanded_df.iloc[split[1]]

            train_dict = {'Phrase': train_df['Phrase'].to_list(),
                          'Sentiment': train_df['Sentiment'].to_list(),
                          'OriginalSentence': train_df['OriginalSentence'].to_list()}

            validation_dict = {'Phrase': validation_df['Phrase'].to_list(),
                               'Sentiment': validation_df['Sentiment'].to_list(),
                               'OriginalSentence': validation_df['OriginalSentence'].to_list()}

            train_dataset = datasets.Dataset.from_dict(train_dict)
            validation_dataset = datasets.Dataset.from_dict(validation_dict)

            dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                                 "validation": validation_dataset})

            datasets_list.append(dataset_dict)

        return datasets_list

    def _format_dataset(self, dataset_tokenized: datasets.DatasetDict) -> datasets.DatasetDict:
        dataset_formatted = dataset_tokenized.remove_columns(['Phrase', 'OriginalSentence'])
        dataset_formatted = dataset_formatted.rename_column('Sentiment', 'label')

        return dataset_formatted


class CustomTrainValHO(CustomDataset):
    def __init__(self, tsv_path: str, cut: int = None, train_set_size: float = 0.8):
        self.train_set_size = train_set_size

        super().__init__(tsv_path, cut)


    def _build_dataset(self, df: pd.DataFrame, cut: int) -> List[datasets.DatasetDict]:
        all_original_sentences = df.groupby(by='SentenceId', as_index=False).first()
        all_original_sentences.drop(columns=['PhraseId', 'Sentiment'], inplace=True)
        all_original_sentences.rename(columns={'Phrase': 'OriginalSentence'}, inplace=True)

        expanded_df = df.merge(all_original_sentences, on='SentenceId')[:cut]

        train_df, validation_df = train_test_split(expanded_df,
                                                   train_size=self.train_set_size,
                                                   stratify=expanded_df['Sentiment'],
                                                   shuffle=True)

        train_dict = {'Phrase': train_df['Phrase'].to_list(),
                      'Sentiment': train_df['Sentiment'].to_list(),
                      'OriginalSentence': train_df['OriginalSentence'].to_list()}

        validation_dict = {'Phrase': validation_df['Phrase'].to_list(),
                           'Sentiment': validation_df['Sentiment'].to_list(),
                           'OriginalSentence': validation_df['OriginalSentence'].to_list()}

        train_dataset = datasets.Dataset.from_dict(train_dict)
        validation_dataset = datasets.Dataset.from_dict(validation_dict)

        dataset_dict = datasets.DatasetDict({"train": train_dataset,
                                             "validation": validation_dataset})

        return [dataset_dict]

    def _format_dataset(self, dataset_tokenized: datasets.DatasetDict) -> datasets.DatasetDict:
        dataset_formatted = dataset_tokenized.remove_columns(['Phrase', 'OriginalSentence'])
        dataset_formatted = dataset_formatted.rename_column('Sentiment', 'label')

        return dataset_formatted


class CustomTest(CustomDataset):

    def _build_dataset(self, df: pd.DataFrame, cut: int) -> List[datasets.DatasetDict]:
        all_original_sentences = df.groupby(by='SentenceId', as_index=False).first()
        all_original_sentences.drop(columns=['PhraseId'], inplace=True)
        all_original_sentences.rename(columns={'Phrase': 'OriginalSentence'}, inplace=True)

        expanded_df = df.merge(all_original_sentences, on='SentenceId')[:cut]

        test_dict = {'PhraseId': expanded_df['PhraseId'].to_list(),
                     'Phrase': expanded_df['Phrase'].to_list(),
                     'OriginalSentence': expanded_df['OriginalSentence'].to_list()}

        test_dataset = datasets.Dataset.from_dict(test_dict)

        dataset_dict = datasets.DatasetDict({"test": test_dataset})

        return [dataset_dict]

    def _format_dataset(self, dataset_tokenized: datasets.DatasetDict) -> datasets.DatasetDict:
        return dataset_tokenized.remove_columns(['Phrase', 'OriginalSentence'])
