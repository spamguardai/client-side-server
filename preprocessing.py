import numpy as np
from collections import Counter
from config import MAX_FEATURES
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Uncomment these lines to download necessary NLTK data if not already installed
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TfidfVectorizerWrapper:
    def __init__(self, max_features=MAX_FEATURES, ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts).toarray()
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

class SimpleCountVectorizer:
    def __init__(self, max_features=MAX_FEATURES):
        self.vocabulary = {}

    def fit(self, texts):
        words = set()
        for text in texts:
            if isinstance(text, str):
                words.update(self._tokenize(text))
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(words))}
        return self

    def transform(self, texts):
        vectors = np.zeros((len(texts), len(self.vocabulary)))
        for i, text in enumerate(texts):
            if isinstance(text, str):
                tokens = self._tokenize(text)
                counts = Counter(tokens)
                for word, count in counts.items():
                    if word in self.vocabulary:
                        vectors[i][self.vocabulary[word]] = count
        return vectors

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def _tokenize(self, text):
        return text.lower().split() if isinstance(text, str) else []
    
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return " ".join(words)

def vectorize(texts, vectorizer, important_indices=None, fit_vectorizer=False):
    # 1. Fit or transform the input texts
    if fit_vectorizer:
        X = vectorizer.fit_transform(texts)
        X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)
        
        # 2. Calculate variance to pick important features
        feature_variance = np.var(X_dense, axis=0)
        important_indices = np.argsort(-feature_variance)[:MAX_FEATURES]
    else:
        X = vectorizer.transform(texts)
        X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)

    # 3. Select top features only
    X_selected = X_dense[:, important_indices]

    return X_selected, important_indices
