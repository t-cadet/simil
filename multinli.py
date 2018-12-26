import os

# ngram_vectorize
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

# model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import models, layers, initializers, regularizers
from tensorflow.python.keras.layers import Dense, Dropout, Input, Embedding, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D

#train_model
import tensorflow as tf

import logging
import json
import random

import numpy as np

# tensorboard
from tensorflow.python.keras.callbacks import TensorBoard
from time import time

# sequence vectorize
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text

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

def sequence_vectorize(pre, hyp, top_k=20000, max_seq_len=500):
    """Vectorizes texts as sequence vectors."""

    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=top_k)
    tokenizer.fit_on_texts(pre+hyp)

    # Vectorize training texts.
    x_hyp = tokenizer.texts_to_sequences(pre)
    x_pre = tokenizer.texts_to_sequences(hyp)

    # Get max sequence length.
    max_length = len(max(x_hyp+x_pre, key=len))
    if max_length > max_seq_len:
        max_length = max_seq_len

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_hyp = sequence.pad_sequences(x_hyp, maxlen=max_length).astype('float32')
    x_pre = sequence.pad_sequences(x_pre, maxlen=max_length).astype('float32')
    return x_hyp, x_pre, tokenizer

# def sepcnn_model(input_shape,
#                  num_features,
#                  blocks=2,
#                  filters=64,
#                  kernel_size=3,
#                  embedding_dim=200,
#                  dropout_rate=0.2,
#                  pool_size=3,
#                  num_classes=3,
#                  optimizer='rmsprop',
#                  use_pretrained_embedding=False,
#                  is_embedding_trainable=False,
#                  embedding_matrix=None):
#     """Creates and compiles an instance of a separable CNN model.

#     # Arguments
#         blocks: int, number of pairs of sepCNN and pooling blocks in the model.
#         filters: int, output dimension of the layers.
#         kernel_size: int, length of the convolution window.
#         embedding_dim: int, dimension of the embedding vectors.
#         dropout_rate: float, percentage of input to drop at Dropout layers.
#         pool_size: int, factor by which to downscale input at MaxPooling layer.
#         input_shape: tuple, shape of input to the model.
#         num_classes: int, number of output classes.
#         num_features: int, number of words (embedding input dimension).
#         use_pretrained_embedding: bool, true if pre-trained embedding is on.
#         is_embedding_trainable: bool, true if embedding layer is trainable.
#         embedding_matrix: dict, dictionary with embedding coefficients.

#     # Returns
#         A compiled sepCNN model instance.
#     """
#     def TwoSepC1D(inp, filters=filters, kernel_size=kernel_size):
#         out = SeparableConv1D(filters=filters,
#                                   kernel_size=kernel_size,
#                                   activation='relu',
#                                   bias_initializer='random_uniform',
#                                   depthwise_initializer='random_uniform',
#                                   padding='same')(inp)
#         out = SeparableConv1D(filters=filters,
#                                   kernel_size=kernel_size,
#                                   activation='relu',
#                                   bias_initializer='random_uniform',
#                                   depthwise_initializer='random_uniform',
#                                   padding='same')(out)
#         return out

#     # Add embedding layer. If pre-trained embedding is used add weights to the
#     # embeddings layer and set trainable to input is_embedding_trainable flag.
#     if use_pretrained_embedding:
#         embedding_layer = Embedding(input_dim=num_features,
#                             output_dim=embedding_dim,
#                             input_length=input_shape[0],
#                             weights=[embedding_matrix],
#                             trainable=is_embedding_trainable)
#     else:
#         embedding_layer = Embedding(input_dim=num_features,
#                             output_dim=embedding_dim,
#                             input_length=input_shape[0])

#     pre_input = Input(shape=input_shape)
#     hyp_input = Input(shape=input_shape)

#     pre_out = embedding_layer(pre_input)
#     hyp_out = embedding_layer(hyp_input)

#     for _ in range(blocks-1):
#         pre_out = Dropout(rate=dropout_rate)(pre_out)
#         pre_out = TwoSepC1D(pre_out)
#         pre_out = MaxPooling1D(pool_size=pool_size)(pre_out)

#         hyp_out = Dropout(rate=dropout_rate)(hyp_out)
#         hyp_out = TwoSepC1D(hyp_out)
#         hyp_out = MaxPooling1D(pool_size=pool_size)(hyp_out)

#     pre_out = TwoSepC1D(pre_out, filters=filters*2)
#     pre_out = GlobalAveragePooling1D()(pre_out)
#     pre_out = Dropout(rate=dropout_rate)(pre_out)

#     hyp_out = TwoSepC1D(hyp_out, filters=filters*2)
#     hyp_out = GlobalAveragePooling1D()(hyp_out)
#     hyp_out = Dropout(rate=dropout_rate)(hyp_out)

#     out = layers.concatenate([pre_out, hyp_out], axis=-1)
#     out = Dense(32, 'relu')(out)
#     op_units, op_activation, loss = (1, 'sigmoid', 'binary_crossentropy') if num_classes==2 else (num_classes, 'softmax', 'sparse_categorical_crossentropy')
#     out = Dense(op_units, activation=op_activation)(out)

#     model = Model([pre_input, hyp_input], out)
#     model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
#     return model

def sepcnn_model(input_shape,
                 num_features,
                 blocks=2,
                 filters=64,
                 kernel_size=3,
                 embedding_dim=200,
                 dropout_rate=0.2,
                 pool_size=3,
                 num_classes=3,
                 optimizer='rmsprop',
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):
    """Creates and compiles an instance of a separable CNN model.

    # Arguments
        blocks: int, number of pairs of sepCNN and pooling blocks in the model.
        filters: int, output dimension of the layers.
        kernel_size: int, length of the convolution window.
        embedding_dim: int, dimension of the embedding vectors.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        pool_size: int, factor by which to downscale input at MaxPooling layer.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.
        num_features: int, number of words (embedding input dimension).
        use_pretrained_embedding: bool, true if pre-trained embedding is on.
        is_embedding_trainable: bool, true if embedding layer is trainable.
        embedding_matrix: dict, dictionary with embedding coefficients.

    # Returns
        A compiled sepCNN model instance.
    """
    def TwoSepC1D(inp, filters=filters, kernel_size=kernel_size):
        out = SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same')(inp)
        out = SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same')(out)
        return out

    # Add embedding layer. If pre-trained embedding is used add weights to the
    # embeddings layer and set trainable to input is_embedding_trainable flag.
    if use_pretrained_embedding:
        embedding_layer = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable)
    else:
        embedding_layer = Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0])

    inp = Input(shape=input_shape)
    x_out = embedding_layer(inp)
    for _ in range(blocks-1):
        x_out = Dropout(rate=dropout_rate)(x_out)
        x_out = TwoSepC1D(x_out)
        x_out = MaxPooling1D(pool_size=pool_size)(x_out)
    x_out = TwoSepC1D(x_out, filters=filters*2)
    x_out = GlobalAveragePooling1D()(x_out)
    x_out = Dropout(rate=dropout_rate)(x_out)

    encoder = Model(inp, x_out)


    pre_input = Input(shape=input_shape)
    hyp_input = Input(shape=input_shape)

    pre_out = encoder(pre_input)
    hyp_out = encoder(hyp_input)

    out = layers.concatenate([pre_out, hyp_out], axis=-1)
    out = Dense(32, 'relu')(out)
    op_units, op_activation, loss = (1, 'sigmoid', 'binary_crossentropy') if num_classes==2 else (num_classes, 'softmax', 'sparse_categorical_crossentropy')
    out = Dense(op_units, activation=op_activation)(out)

    model = Model([pre_input, hyp_input], out)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
    return model

def train_model(model, x_train, train_labels, epochs=1000, val_split=0.1, batch_size=32, filename='mnli_mlp_model', tensorboard=False):

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
    if tensorboard:
        callbacks.append(TensorBoard(log_dir="tensorboard/mnli/{}".format(time()), histogram_freq=1, batch_size=32, write_graph=True))

    # Train and validate model.
    history = model.fit(
            x_train,
            train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=val_split,
            verbose=1,
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

def runBow():
    t_pre, t_hyp, t_lab = load_mnli(lim=15000)
    pre_train, hyp_train, v, s = ngram_vectorize(t_pre, t_hyp, t_lab)

    model = siamese_mlp_model(units=[8], input_shape=pre_train.shape[1:], num_classes=3, optimizer=tf.keras.optimizers.Adam(lr=1e-3))
    train_model(model, [pre_train, hyp_train], t_lab, epochs=3, tensorboard=True)

TOP_K = 20000
t_pre, t_hyp, t_lab = load_mnli(lim=10000)
pre_train, hyp_train, t = sequence_vectorize(t_pre, t_hyp, TOP_K)

model = sepcnn_model(input_shape=pre_train.shape[1:],
                     num_features=min(len(t.word_index) + 1, TOP_K),
                     blocks=2,
                     filters=64,
                     kernel_size=5,
                     embedding_dim=50,
                     dropout_rate=0.2,
                     pool_size=3,
                     num_classes=3,
                     optimizer=tf.keras.optimizers.Adam(lr=1e-3))

train_model(model, [pre_train, hyp_train], t_lab, epochs=1000, tensorboard=False, batch_size=512, filename='mnli_sepcnn_model')
