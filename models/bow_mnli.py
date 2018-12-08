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

@Model.register("bow_mnli")
class BowMNLI(Model):
    def __init__(self, vocab: Vocabulary,
                premise_encoder: Optional[Seq2SeqEncoder] = None,
                hypothesis_encoder: Optional[Seq2SeqEncoder] = None,
                initializer: InitializerApplicator = InitializerApplicator(),
                regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BowMNLI, self).__init__(vocab, regularizer)
        
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