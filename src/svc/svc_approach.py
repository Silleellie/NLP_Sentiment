import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import spacy

class SVCApproach:

    def __init__(self, model = 'en_core_web_sm', **args):
        if model not in spacy.cli.info()['pipelines']:
            spacy.cli.download(model)

        self.nlp = spacy.load(model)
        self.nlp.add_pipe('merge_entities')

        self.vectorizer = CountVectorizer(binary=True, analyzer=self.preprocessing)

        self.svc = SVC(**args)

    def preprocessing(self, phrase: str):
        phrase_data = list(self.nlp(phrase))
        token_list = []
        for w in phrase_data:
            if not w.ent_type_:
                token_list.append(w.lemma_.lower())
            else:
                token_list.append('<' + w.ent_type_ + '>')
        return token_list
    
    def train(self, train_texts, train_labels):
        train_vectors = self.vectorizer.fit_transform(train_texts)
        self.svc.fit(train_vectors, train_labels)
    
    def predict(self, test_texts):
        return self.svc.predict(self.vectorizer.transform(test_texts))
    
    def test_and_submit(self, test_texts, test_ids):
        predictions = self.predict(test_texts)
        columns = {'PhraseId': test_ids, 'Sentiment': predictions}
        df = pd.DataFrame(data=columns)
        df.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    train = pd.read_csv("../../dataset/train.tsv", sep="\t")
    test = pd.read_csv("../../dataset/test.tsv", sep="\t")
    X_train = train['Phrase'].tolist()
    Y_train = train['Sentiment'].tolist()
    X_test = test['Phrase'].tolist()
    phrase_ids = test['PhraseId'].tolist()

    svc = SVCApproach()
    svc.train(X_train, Y_train)
    svc.test_and_submit(X_test, phrase_ids)
