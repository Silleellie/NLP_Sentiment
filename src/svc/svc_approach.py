import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import spacy

model = 'en_core_web_sm'

if model not in spacy.cli.info()['pipelines']:
    spacy.cli.download(model)

nlp = spacy.load(model)
nlp.add_pipe('merge_entities')


def preprocessing(phrase: str):
    phrase_data = list(nlp(phrase))
    token_list = []
    for w in phrase_data:
        if not w.ent_type_:
            token_list.append(w.lemma_.lower())
        else:
            token_list.append('<' + w.ent_type_ + '>')
    return token_list


if __name__ == '__main__':
    train = pd.read_csv("../../dataset/train.tsv", sep="\t")
    test = pd.read_csv("../../dataset/test.tsv", sep="\t")
    X_train = train['Phrase'].tolist()
    Y_train = train['Sentiment'].tolist()
    X_test = test['Phrase'].tolist()
    phrase_id = test['PhraseId'].tolist()

    vectorizer = CountVectorizer(binary=True, analyzer=preprocessing)
    X_train = vectorizer.fit_transform(X_train)

    classif = SVC()
    classif.fit(X_train, Y_train)

    Y_pred = []

    for phrase, x in zip(X_test, classif.predict(vectorizer.transform(X_test))):
        print("SCORE: ", x)
        print("PHRASE: ", phrase)
        Y_pred.append(x)

    columns = {'PhraseId': phrase_id, 'Sentiment': Y_pred}
    df = pd.DataFrame(data=columns)
    df.to_csv("submission.csv", index=False)
