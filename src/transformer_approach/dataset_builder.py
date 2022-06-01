import datasets
import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
from sklearn.model_selection import train_test_split


class CustomDataset:

    def __init__(self, tsv_path: str):
        df = pd.read_csv(tsv_path, sep='\t')

        all_original_sentences = df.groupby(by='SentenceId', as_index=False).first()
        all_original_sentences.drop(columns=['PhraseId', 'Sentiment'], inplace=True)
        all_original_sentences.rename(columns={'Phrase': 'OriginalSentence'}, inplace=True)

        expanded_df = df.merge(all_original_sentences, on='SentenceId')

        self.dataset_dict = self._build_dataset(expanded_df)
        # load tagger
        self.tagger = SequenceTagger.load("flair/pos-english-fast")

    @staticmethod
    def _build_dataset(df: pd.DataFrame) -> datasets.DatasetDict:

        train_df, validation_df = train_test_split(df,
                                                   train_size=0.8,
                                                   stratify=df['Sentiment'],
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
        dataset_formatted = None
        if mode == 'only_phrase':

            dataset_tokenized = self.dataset_dict.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item),
                                                      batched=True)

            dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label').remove_columns('Phrase')
        elif mode == 'with_pos':

            dataset_pos = self.dataset_dict.map(lambda batch_item: self.pos_tagger_fn(batch_item))
            dataset_tokenized = dataset_pos.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item,
                                                                                     field_2="Pos"),
                                                batched=True)

            dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label').remove_columns(['Phrase', 'Pos'])
        elif mode == 'with_reference':
            dataset_tokenized = self.dataset_dict.map(lambda batch_item: self.tokenizer_fn(tokenizer, batch_item,
                                                                                           field_2="OriginalSentence"),
                                                      batched=True)

            dataset_formatted = dataset_tokenized.rename_column('Sentiment', 'label')
            dataset_formatted = dataset_formatted.remove_columns(['Phrase', 'OriginalSentence'])

        return dataset_formatted

    def tokenizer_fn(self, tokenizer, batch_item, field_2=None):
        if field_2 is not None:
            return tokenizer(batch_item["Phrase"], batch_item[field_2], truncation=True)

        return tokenizer(batch_item['Phrase'])

    def pos_tagger_fn(self, batch_item):
        # make example sentence
        sentence = Sentence(batch_item["Phrase"])

        # predict NER tags
        self.tagger.predict(sentence)

        # print sentence
        tags = ' '.join(token.tag for token in sentence.tokens)

        return {'Pos': tags}
