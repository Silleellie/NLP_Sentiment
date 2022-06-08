import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
            if not w.is_punct:
                token_list.append(w.lemma_.lower())
        else:
            token_list.append('<' + w.ent_type_ + '>')
    return token_list


train = pd.read_csv("train.tsv", sep="\t")
# test = pd.read_csv("test.tsv", sep="\t")
X = train['Phrase'].tolist()
Y = train['Sentiment'].tolist()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
phrase_id = train['PhraseId'].tolist()[-len(Y_test):]

vectorizer = CountVectorizer(binary=True, analyzer=preprocessing)
X_train = vectorizer.fit_transform([phrase for phrase in X_train])

classif = SVC()
classif.fit(X_train, Y_train)

Y_pred = []

for phrase, x in zip(X_test, classif.predict(vectorizer.transform([phrase for phrase in X_test]))):
    print("SCORE: ", x)
    print("PHRASE: ", phrase)
    Y_pred.append(x)

print(accuracy_score(Y_test, Y_pred))

columns = {'PhraseId': phrase_id, 'Sentiment': Y_pred}
df = pd.DataFrame(data=columns)
df.to_csv("output.csv", index=False)
