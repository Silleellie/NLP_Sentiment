from abc import abstractmethod

import datasets
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.model_selection import train_test_split

# load tagger as class attribute
tagger = SequenceTagger.load("flair/pos-english-fast")

class CustomDataset:
    def __init__(self, tsv_path: str, cut: int = None):
        df = pd.read_csv(tsv_path, sep='\t')

        self.dataset_dict = self._build_dataset(df, cut)

    @staticmethod
    @abstractmethod
    def _build_dataset(df: pd.DataFrame, cut: int) -> datasets.DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, tokenizer, mode='only_phrase') -> datasets.DatasetDict:
        dataset_tokenized = None
        if mode == 'only_phrase':

            dataset_tokenized = self.dataset_dict.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item),
                                                      batched=True)
        elif mode == 'with_pos':

            dataset_pos = self.dataset_dict.map(lambda batch_item: self.pos_tagger_fn(batch_item))
            dataset_tokenized = dataset_pos.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item,
                                                                                     field_2="Pos"),
                                                batched=True)
            dataset_tokenized = dataset_tokenized.remove_columns('Pos')
        elif mode == 'with_reference':
            dataset_tokenized = self.dataset_dict.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item,
                                                                                           field_2="OriginalSentence"),
                                                      batched=True)

        dataset_formatted = dataset_tokenized.remove_columns(['Phrase', 'OriginalSentence'])

        return dataset_formatted

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


class CustomTrainVal(CustomDataset):
    @staticmethod
    def _build_dataset(df: pd.DataFrame, cut: int) -> datasets.DatasetDict:
        all_original_sentences = df.groupby(by='SentenceId', as_index=False).first()
        all_original_sentences.drop(columns=['PhraseId', 'Sentiment'], inplace=True)
        all_original_sentences.rename(columns={'Phrase': 'OriginalSentence'}, inplace=True)

        expanded_df = df.merge(all_original_sentences, on='SentenceId')[:cut]

        train_df, validation_df = train_test_split(expanded_df,
                                                   train_size=0.8,
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

        return dataset_dict

    def preprocess(self, tokenizer, mode='only_phrase') -> datasets.DatasetDict:
        dataset_tokenized = super().preprocess(tokenizer, mode)

        dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label')

        return dataset_formatted


class CustomTest(CustomDataset):
    @staticmethod
    def _build_dataset(df: pd.DataFrame, cut: int) -> datasets.DatasetDict:
        all_original_sentences = df.groupby(by='SentenceId', as_index=False).first()
        all_original_sentences.drop(columns=['PhraseId'], inplace=True)
        all_original_sentences.rename(columns={'Phrase': 'OriginalSentence'}, inplace=True)

        expanded_df = df.merge(all_original_sentences, on='SentenceId')[:cut]

        test_dict = {'PhraseId': expanded_df['PhraseId'].to_list(),
                     'Phrase': expanded_df['Phrase'].to_list(),
                     'OriginalSentence': expanded_df['OriginalSentence'].to_list()}

        test_dataset = datasets.Dataset.from_dict(test_dict)

        dataset_dict = datasets.DatasetDict({"test": test_dataset})

        return dataset_dict

    def preprocess(self, tokenizer, mode='only_phrase') -> datasets.DatasetDict:
        return super().preprocess(tokenizer, mode)
