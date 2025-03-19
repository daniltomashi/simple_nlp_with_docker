import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from text_processing import TextProcessing
import pickle
import os




class TraditionalMl:
    """
        Takes model, data and vectorizer as input. Class as interface for model and data manupulation
    """
    def __init__(self, df:pd.DataFrame, text_name, target_name, model, vectorizer, label_encoder, metrics, process_text=True):
        self.df = df
        self.text_name = text_name
        self.target_name = target_name
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.metrics = metrics
        self.to_process_text = process_text

    def process_text(self):
        text_processor = TextProcessing()
        self.df[text_name] = text_processor.preprocess(self.df[text_name])

    def process_target(self):
        self.df[target_name] = self.label_encoder.fit_transform(self.df[target_name])

    def split_data(self, test_size=0.15):
        X_train, X_test, y_train, y_test = train_test_split(self.df[self.text_name], self.df[self.target_name], 
                                                            test_size=test_size)
        
        return X_train, X_test, y_train, y_test

    def get_metrics(self, y_pred, y):
        return {metric.__name__:metric(y_pred, y) for metric in self.metrics}

    def train_model(self):
        if self.to_process_text:
            self.process_text()
        self.process_target()

        X_train, X_test, y_train, y_test = self.split_data()
        # fit vectorizer with only train data
        self.vectorizer.fit(X_train)
        
        # transform data before training and train data
        X_train, X_test = self.vectorizer.transform(X_train), self.vectorizer.transform(X_test)
        self.model.fit(X_train, y_train)

        # test data
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        print("Train metrics -->", self.get_metrics(y_train_pred, y_train))
        print("Test metrics -->", self.get_metrics(y_test_pred, y_test))


    def get_model(self):
        return self.model

    def get_vectorizer(self):
        return self.vectorizer
    
    def get_label_encoder(self):
        return self.label_encoder


# training process
if __name__ == "__main__":
    file = [i for i in os.listdir('data/') if '.csv' in i][0]
    df = pd.read_csv('data/'+file)
    
    text_name = "Message"
    target_name = "Category"

    model = LogisticRegression()
    vectorizer = TfidfVectorizer()
    label_encoder = LabelEncoder()
    metrics = [accuracy_score, f1_score]


    approach = TraditionalMl(df.copy(), text_name, target_name, model, vectorizer, label_encoder, metrics)
    approach.train_model()

    if "tools" not in os.listdir():
        os.mkdir("tools")

    with open("tools/label_encoder.pkl", "wb") as f:
        pickle.dump(approach.get_label_encoder(), f)

    with open("tools/vectorizer.pkl", "wb") as f:
        pickle.dump(approach.get_vectorizer(), f)

    with open("tools/model.pkl", "wb") as f:
        pickle.dump(approach.get_model(), f)
