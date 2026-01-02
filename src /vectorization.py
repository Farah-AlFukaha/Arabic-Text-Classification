import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

#  Bag-of-Words 
def bow_vectorize(texts, max_features=5000):
    vec = CountVectorizer(max_features=max_features, token_pattern=r"(?u)\b[^\d\W]+\b", ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return X, vec

#  TF-IDF
def tfidf_vectorize(texts, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"(?u)\b[^\d\W]+\b")
    X = vec.fit_transform(texts)
    return X, vec

#  Word2Vec 
def word2vec_vectorize(texts, size=300, window=5, min_count=1, epochs=30):
    tokenized_docs = [word_tokenize(doc) for doc in texts]
    w2v_model = Word2Vec(sentences=tokenized_docs,
                         vector_size=size,
                         window=window,
                         min_count=min_count,
                         sg=1,
                         epochs=epochs)

    def get_avg_vector(tokens, model):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if not vectors:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    X = np.array([get_avg_vector(tokens, w2v_model) for tokens in tokenized_docs])
    return X, w2v_model
