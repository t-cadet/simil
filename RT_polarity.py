import os
import random
import numpy as np

from sklearn.model_selection import train_test_split

def load_rt_polarity_dataset(data_path = "datasets", seed=123, test_split=0.15):
    """Loads the rt-polarity dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and test data.
        Number of training samples: 10662
        Number of categories: 2 (0 - negative, 1 - positive)

    # References
    v1.0 sentence polarity dataset comes
    from the URL
        Bo Pang and Lillian Lee,
        'Seeing stars: Exploiting class relationships for sentiment categorization
        with respect to rating scales.', Proceedings of the ACL, 2005.

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

X_train, X_test, y_train, y_test = load_rt_polarity_dataset()
