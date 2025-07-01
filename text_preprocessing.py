import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class TextPreprocessor:
    def __init__(self, normalization_file=None):
        self.normalization_dict = {}
        if normalization_file:
            normalization_df = pd.read_csv(normalization_file)
            self.normalization_dict = normalization_df.set_index('before')['after'].to_dict()
        
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        
        self.stopwords = set(stopwords.words('indonesian'))
        self.negation_words = {
            'tidak', 'bukan', 'belum', 'tak', 'jangan', 'ga', 'gak', 'nggak', 'enggak', 'ndak',
            'kurang', 'tiada'
        }


    def case_folding(self, text):
        text = text.lower()  
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
        text = re.sub(r'\s+', ' ', text).strip()  
        return text

    def tokenization(self, text):
        return word_tokenize(text)

    def normalize(self, words):
        return [self.normalization_dict.get(word, word) for word in words]

    def remove_stopwords(self, words):
        stopwords_tambahan = {'ya'}
        all_stopwords = self.stopwords.union(stopwords_tambahan)
        # Jangan hapus kata negasi
        all_stopwords = all_stopwords.difference(self.negation_words)
        return [word for word in words if word not in all_stopwords]


    def handle_negation(self, words):
        negation_flag = False
        processed_words = []
        
        for word in words:
            if word in self.negation_words:
                negation_flag = True
            elif negation_flag:
                processed_words.append(word + '_NEG')
                negation_flag = False
            else:
                processed_words.append(word)
        
        return processed_words

    def stemming(self, words):
        return [self.stemmer.stem(word.replace('_NEG', '')) + ('_NEG' if '_NEG' in word else '') for word in words]
    
    def preprocess(self, text):
        text = self.case_folding(text)
        tokens = self.tokenization(text)
        tokens = self.normalize(tokens)
        tokens = self.handle_negation(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stemming(tokens)
        return tokens
