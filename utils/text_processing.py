import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk




# download all necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download("punkt_tab")


class TextProcessing:
    """
        Preprocess text from pandas Series
    """
    def __init__(self, to_lemmatize=True):
        self.to_lemmatize = to_lemmatize

        if to_lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
    def lowercasing(self, data):
        return data.apply(lambda x: x.lower())

    def remove_punctuation(self, data):
        return data.apply(lambda x: re.sub(f"[{string.punctuation}]", "", x))

    def remove_spec_symbols(self, data):
        data = data.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
        data = data.apply(lambda x: re.sub(r'\s+', ' ', x))

        return data

    def tokenize(self, data):
        return data.apply(lambda x: word_tokenize(x))

    def lemmatize(self, data):
        return data.apply(lambda x: [self.lemmatizer.lemmatize(word) for word in x])

    def remove_stopwords(self, data):
        stop_words = set(stopwords.words('english'))
        
        return data.apply(lambda x: [word for word in x if word not in stop_words])
        
    
    def preprocess(self, data, to_lowercase=True, to_remove_punctuation=True, to_remove_stopwords=True, 
                   to_remove_spec_symbols=True):
        if to_lowercase:
            data = self.lowercasing(data)

        if to_remove_punctuation:
            data = self.remove_punctuation(data)

        if to_remove_spec_symbols:
            data = self.remove_spec_symbols(data)

        data = self.tokenize(data)

        if to_remove_stopwords:
            data = self.remove_stopwords(data)


        if self.to_lemmatize:
            data = self.lemmatize(data)

        # join all data
        data = data.apply(lambda x: ' '.join(x))

        return data