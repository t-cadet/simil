import numpy as np
from allennlp.data.dataset_readers import SnliReader
from allennlp.data.vocabulary import Vocabulary

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras import layers

def getMnliBow(dataset, vocab):

    def toBow(sentence, vocab):
        bow = np.zeros(vocab.get_vocab_size())
        bow[[vocab.get_token_index(token.text) for token in sentence]] = 1
        return bow

    premises = np.zeros((len(dataset), vocab.get_vocab_size()))
    hypothesis = np.zeros((len(dataset), vocab.get_vocab_size()))
    labels = np.zeros((len(dataset), vocab.get_vocab_size(namespace='labels')))

    i = 0
    for instance in dataset:
        premises[i] = toBow(instance.fields['premise'].tokens, vocab)
        hypothesis[i] = toBow(instance.fields['hypothesis'].tokens, vocab)
        labels[i, vocab.get_token_index(instance.fields['label'].label, namespace="labels")] = 1 
        i = i+1
    return (premises, hypothesis, labels)

reader = SnliReader()

# train_dataset = reader.read(cached_path('datasets/multinli_1.0/multinli_1.0_train.jsonl'))
train_dataset = reader.read('tests/fixtures/train1000.jsonl') # Fixture
validation_dataset = reader.read('tests/fixtures/val1000.jsonl') # Fixture
#validation_dataset = reader.read('datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl')

# print(train_dataset)

vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# vocab.print_statistics()

t_premises, t_hypothesis, t_labels = getMnliBow(train_dataset, vocab)
v_premises, v_hypothesis, v_labels = getMnliBow(validation_dataset, vocab)

# for i in range(3):
#     print(i)
#     print(t_premises[i])
#     print(t_hypothesis[i])
#     print(t_labels[i])

prem_input = Input(shape=(vocab.get_vocab_size('tokens'),))
# prem_out = Dense(32, activation='relu')(prem_input)
# prem_out = Dense(16, activation='relu')(prem_out)
prem_out = Dense(8, activation='hard_sigmoid')(prem_input)

hyp_input = Input(shape=(vocab.get_vocab_size('tokens'),))
# hyp_out = Dense(32, activation='relu')(hyp_input)
# hyp_out = Dense(16, activation='relu')(hyp_out)
hyp_out = Dense(8, activation='hard_sigmoid')(hyp_input)

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
    epochs=100,
    verbose=1,
    validation_data=([v_premises, v_hypothesis], v_labels)
)
