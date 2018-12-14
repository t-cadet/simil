import os

# ngram_vectorize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

# mlp_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.layers import Dense, Dropout, Input

#train_model
import tensorflow as tf

import logging
import json
import random

import numpy as np

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    #filename="log"++".txt"
    )

def load_mnli(lim=None, filename="multinli_1.0_train.jsonl", data_path = "datasets/multinli_1.0", seed=123):
    mnli_path = os.path.join(data_path, filename)
    with open(mnli_path) as mnli_file:
        logging.info("Reading MNLI instances from jsonl dataset at: %s", mnli_path)
        pre, hyp, lab = [], [], []
        for line in mnli_file:
            sample = json.loads(line)
            if sample["gold_label"] == '-':
                continue
            pre.append(sample["sentence1"])
            hyp.append(sample["sentence2"])
            lab.append(sample["gold_label"])
            if lim is not None:
                lim -= 1
                if lim == 0:
                    break

        random.seed(seed)
        random.shuffle(pre)
        random.seed(seed)
        random.shuffle(hyp)
        random.seed(seed)
        random.shuffle(lab)
        lab = preprocessing.LabelEncoder().fit_transform(lab)
        return pre, hyp, lab

def ngram_vectorize(pre, hyp, train_labels, ngram_range=(1,2), top_k=20000, token_mode='word', min_doc_freq=2):
    kwargs = {
            'ngram_range': ngram_range,
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': token_mode,
            'min_df': min_doc_freq,
    }
    vectorizer = TfidfVectorizer(**kwargs)
    vectorizer.fit(pre+hyp)
    pre_train = vectorizer.transform(pre)
    hyp_train = vectorizer.transform(hyp)

    temp = pre_train + hyp_train
    selector = SelectKBest(f_classif, k=min(top_k, temp.shape[1]))
    selector.fit(temp, train_labels)

    pre_train = selector.transform(pre_train).astype('float32')
    hyp_train = selector.transform(hyp_train).astype('float32')
    return pre_train, hyp_train, vectorizer, selector

def siamese_mlp_model(units, input_shape, num_classes = 3, dropout_rate = 0.2, activation='relu', optimizer='rmsprop'):

    prem_input = Input(shape=input_shape)
    hyp_input = Input(shape=input_shape)
    prem_out = Dropout(rate=dropout_rate, input_shape=input_shape)(prem_input)
    hyp_out = Dropout(rate=dropout_rate, input_shape=input_shape)(hyp_input)

    for dim in units:
        prem_out = Dense(dim, activation=activation)(prem_out)
        hyp_out = Dense(dim, activation=activation)(hyp_out)
        prem_out = Dropout(rate=dropout_rate)(prem_out)
        hyp_out = Dropout(rate=dropout_rate)(hyp_out)

    concatenated = layers.concatenate([prem_out, hyp_out], axis=-1)

    op_units, op_activation, loss = (1, 'sigmoid', 'binary_crossentropy') if num_classes==2 else (num_classes, 'softmax', 'sparse_categorical_crossentropy')
    output = Dense(op_units, activation=op_activation)(concatenated)

    model = Model([prem_input, hyp_input], output)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    return model

def train_model(model, x_train, train_labels, epochs=1000, val_split=0.2, batch_size=32, filename='mnli_mlp_model'):

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=val_split,
            verbose=2,  # Logs once per epoch.
            batch_size=batch_size)

    # Print results.
    history = history.history
    logging.info('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('serial/'+filename+'.h5')
    return history['val_acc'][-1], history['val_loss'][-1]

def run_test(model, vectorizer, selector, save_stats=False):
    test_pre_raw, test_hyp_raw, test_lab = load_mnli(filename="multinli_1.0_dev_matched.jsonl")
    test_pre = selector.transform(vectorizer.transform(test_pre_raw))
    test_hyp = selector.transform(vectorizer.transform(test_hyp_raw))
    logging.info(model.evaluate([test_pre,test_hyp], test_lab))

    if save_stats:
        predictions = model.predict([test_pre,test_hyp])
        wrong, right = [], []
        for i in range(len(test_pre_raw)):
            pred = np.argmax(predictions[i])
            if pred==test_lab[i]:
                right.append([test_pre_raw[i], test_hyp_raw[i], str(pred), str(test_lab[i])])
            else:
                wrong.append([test_pre_raw[i], test_hyp_raw[i], str(pred), str(test_lab[i])])
        path = "stats/mnli/"
        with open(path+"right_samples_raw.csv","w") as f:
            f.write("premise\thypothesis\tprediction\tlabel\n")
            f.write("\n".join(["\t".join(e) for e in right]))
        with open(path+"wrong_samples_raw.csv","w") as f:
            f.write("premise\thypothesis\tprediction\tlabel\n")
            f.write("\n".join(["\t".join(e) for e in wrong]))


t_pre, t_hyp, t_lab = load_mnli(lim=15000)
pre_train, hyp_train, v, s = ngram_vectorize(t_pre, t_hyp, t_lab)

model = siamese_mlp_model(units=[8], input_shape=pre_train.shape[1:], num_classes=3, optimizer=tf.keras.optimizers.Adam(lr=1e-3))
#train_model(model, [pre_train, hyp_train], t_lab)


