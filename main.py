# TODO: construire 3 modèles corrects 
# refactoriser pour appeler le model que je veux
# analyser les erreurs
# utiliser le dataset MR, refactoriser, analyser erreurs
# implémenter les embeddings 

import numpy as np
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.vocabulary import Vocabulary

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import layers

def mnliToList(dataset, vocab):
    premises = []
    hypothesis = []
    labels = []

    for instance in dataset:
        premises.append([vocab.get_token_index(token.text) for token in instance.fields['premise'].tokens])
        hypothesis.append([vocab.get_token_index(token.text) for token in instance.fields['hypothesis'].tokens])
        labels.append(vocab.get_token_index(instance.fields['label'].label, namespace="labels"))
    return (premises, hypothesis, labels)


def getBow(dataset, vocab, bow_type='groundBow'):
    def countBow(sentence, vocab):
        bow = np.zeros(vocab.get_vocab_size())
        for token_id in sentence:
            bow[token_id]+=1
        return bow
    def groundBow(sentence, vocab):        
        return [0 if c==0 else 1 for c in countBow(sentence, vocab)]
    def freqBow(sentence, vocab):
        return [c/len(sentence) for c in countBow(sentence, vocab)]            

    bows = np.zeros((len(dataset), vocab.get_vocab_size()))
    i = 0
    for sentence in dataset:
        bows[i] = locals()[bow_type](sentence, vocab)
        i = i+1
    return bows

def getMnliBow(dataset, vocab, bow_type='groundBow'):
    premises, hypothesis, labs = mnliToList(dataset, vocab)

    premises = getBow(premises, vocab, bow_type)
    hypothesis = getBow(hypothesis, vocab, bow_type)

    labels = np.zeros((len(dataset), vocab.get_vocab_size(namespace='labels')))
    for i in range(len(dataset)):
        labels[i, labs[i]] = 1

    return (premises, hypothesis, labels)

reader = SnliReader()

# train_dataset = reader.read(cached_path('datasets/multinli_1.0/multinli_1.0_train.jsonl'))
train_dataset = reader.read('tests/fixtures/train1000.jsonl') # Fixture
validation_dataset = reader.read('tests/fixtures/val1000.jsonl') # Fixture
#validation_dataset = reader.read('datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl')

# print(train_dataset)

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# vocab.print_statistics()

t_premises, t_hypothesis, t_labels = getMnliBow(train_dataset, vocab, 'freqBow')
v_premises, v_hypothesis, v_labels = getMnliBow(validation_dataset, vocab, 'freqBow')

# for i in range(3):
#     print(i)
#     print(t_premises[i])
#     print(t_hypothesis[i])
#     print(t_labels[i])

prem_input = Input(shape=(vocab.get_vocab_size('tokens'),))
prem_out = Dense(32, activation='relu')(prem_input)
prem_out = Dense(16, activation='relu')(prem_out)
# prem_out = Dense(8, activation='hard_sigmoid')(prem_input)

hyp_input = Input(shape=(vocab.get_vocab_size('tokens'),))
hyp_out = Dense(32, activation='relu')(hyp_input)
hyp_out = Dense(16, activation='relu')(hyp_out)
# hyp_out = Dense(8, activation='hard_sigmoid')(hyp_input)

concatenated = layers.concatenate([prem_out, hyp_out], axis=-1)
output = Dense(vocab.get_vocab_size('labels'), activation='softmax')(concatenated)

model = Model([prem_input, hyp_input], output)
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['acc'])

model.fit(
    [t_premises, t_hypothesis],
    t_labels,
    batch_size=32,
    epochs=10,
    verbose=1,
    validation_data=([v_premises, v_hypothesis], v_labels)
)
