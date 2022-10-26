import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class nbModel:
    def __init__(self):
        self.df = pd.read_csv('static/data.csv')
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB(alpha=0.1)

    def training(self):
        vectors = self.vectorizer.fit_transform(self.df.text)
        x_train, x_test, y_train, y_test = train_test_split(vectors, self.df.label_num)
        self.model.fit(x_train,y_train)
        return str(self.model.score(x_test,y_test))
    
    def question(self, msg):
        vectors1=self.vectorizer.transform([msg]).toarray()
        return self.model.predict(vectors1)[0]
    
    def dataInfo(self):
        return str(self.df)