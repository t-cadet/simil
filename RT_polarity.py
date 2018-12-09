# load_rt_polarity_dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ngram_vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def load_rt_polarity_dataset(data_path = "datasets", seed=123, test_split=0.15):
    """Loads the rt-polarity dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.
        test_split: float, proportion of test samples.

    # Returns
        A tuple of training and test data.
        Number of samples: 10662
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
        Bo Pang and Lillian Lee, 'Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.', Proceedings of the ACL, 2005.

        Download and uncompress archive from:
        http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polarydata.tar.gz
    """
    rt_path = os.path.join(data_path, 'rt-polaritydata', 'rt-polaritydata')
    pos_path = os.path.join(rt_path, 'rt-polarity.pos')
    neg_path = os.path.join(rt_path, 'rt-polarity.neg')

    with open(pos_path, encoding="latin-1") as pos_f, open(neg_path, encoding="latin-1") as neg_f:
        pos_data = pos_f.readlines()
        neg_data = neg_f.readlines()
    assert len(pos_data)==len(neg_data)

    data = pos_data + neg_data
    labels = [1]*len(pos_data) + [0]*len(neg_data)

    return train_test_split(data, np.array(labels), random_state=seed, test_size=test_split)

def ngram_vectorize(train_texts, train_labels, test_texts, ngram_range=(1,2), top_k=20000, token_mode='word', min_doc_freq=2):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        test_texts: list, test text strings.

    # Returns
        x_train, x_test: vectorized training and test texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': ngram_range,  # Use 1-grams + 2-grams. Range (inclusive) of n-gram sizes for tokenizing text.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': token_mode,  # Whether text should be split into word or character n-grams. One of 'word', 'char'.
            'min_df': min_doc_freq, # Minimum document/corpus frequency below which a token will be discarded.
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize test texts.
    x_test = vectorizer.transform(test_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    return x_train, x_test

x_train, x_test, y_train, y_test = load_rt_polarity_dataset()
x_train, x_test = ngram_vectorize(x_train, y_train, x_test)
