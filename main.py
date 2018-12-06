from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField, Field, LabelField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

import json
import logging

from overrides import overrides
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

torch.manual_seed(1)
class MNLIReader(DatasetReader):
    """
    Reads a file from the Multi Natural Language Inference (MultiNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  Part of the keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """


    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)


        with open(file_path, 'r') as mnli_file:
            logger.info("Reading MultiNLI instances from jsonl dataset at: %s", file_path)
            i = 0;
            for line in mnli_file:
                i = i + 1
                if(i==1000):
                  break;
                example = json.loads(line)


                label = example["gold_label"]
                if label == '-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue


                premise = example["sentence1"]
                hypothesis = example["sentence2"]


                yield self.text_to_instance(premise, hypothesis, label)


    @overrides
    def text_to_instance(self,  # type: ignore
                         premise: str,
                         hypothesis: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields['premise'] = TextField(premise_tokens, self._token_indexers)
        fields['hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)


        metadata = {"premise_tokens": [x.text for x in premise_tokens],
                    "hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

@Model.register("bow")
class Bow(Model):
    def __init__(self, vocab: Vocabulary,
                premise_encoder: Optional[Seq2SeqEncoder] = None,
                hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                initializer: InitializerApplicator = InitializerApplicator(),
                regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Bow, self).__init__(vocab, regularizer)
        
        self._premise_encoder = premise_encoder
        self._hypothesis_encoder = hypothesis_encoder or premise_encoder
        self.fc = nn.Sequential(
            nn.Linear(vocab.get_vocab_size('tokens'), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.aggregate = nn.Sequential(nn.Linear(16, vocab.get_vocab_size('labels')))

        self._num_labels = vocab.get_vocab_size(namespace="labels")
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(self,  # type: ignore
                    premise: Dict[str, torch.LongTensor],
                    hypothesis: Dict[str, torch.LongTensor],
                    label: torch.IntTensor = None,
                    metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()
        
        out1 = self.fc(self._premise_encoder(premise, premise_mask))
        out2 = self.fc(self._hypothesis_encoder(hypothesis, hypothesis_mask))

        label_logits = self.aggregate(torch.cat([out1, out2], dim=-1))
        label_prob = torch.nn.functional.softmax(label_logits, dim=-1) 

        output_dict = {"label_logits": label_logits,
                       "label_probs": label_probs}

        if label is not None:
            output_dict["loss"] = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)

        if metadata is not None:
            output_dict["premise_tokens"] = [x["premise_tokens"] for x in metadata]
            output_dict["hypothesis_tokens"] = [x["hypothesis_tokens"] for x in metadata]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset), }

class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

reader = MNLIReader()
train_dataset = reader.read(cached_path('datasets/multinli_1.0/multinli_1.0_train.jsonl'))
validation_dataset = reader.read(cached_path('datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl'))
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

iterator = BucketIterator(sorting_keys=[("premise", "num_tokens"), ("hypothesis", "num_tokens"), ("premise", "num_token_characters")])
iterator.index_with(vocab)
# EMBEDDING_DIM = 6
# HIDDEN_DIM = 6
# token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                             embedding_dim=EMBEDDING_DIM)
# word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
# lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
# model = LstmTagger(word_embeddings, lstm, vocab)
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# trainer = Trainer(model=model,
#                   optimizer=optimizer,
#                   iterator=iterator,
#                   train_dataset=train_dataset,
#                   validation_dataset=validation_dataset,
#                   patience=10,
#                   num_epochs=1000)
# trainer.train()
# predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
# tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
# tag_ids = np.argmax(tag_logits, axis=-1)
# print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])