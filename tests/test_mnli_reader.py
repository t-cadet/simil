# pylint: disable=no-self-use,invalid-name
import pytest

from allennlp.data.dataset_readers import SnliReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestMnliReader():
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = SnliReader(lazy=lazy)
        instances = reader.read('tests/fixtures/multinli_1.0_train.jsonl')
        instances = ensure_list(instances)

        instance0 = {"premise": ["Conceptually", "cream", "skimming", "has", "two", "basic", "dimensions", "-", "product", "and", "geography", "."],
        			 "hypothesis": ["Product", "and", "geography", "are", "what", "make", "cream", "skimming", "work", "."],
        			 "label": "neutral"}

        instance1 = {"premise": ["you", "know", "during", "the", "season", "and", "i", "guess", "at", "at", "your", "level", "uh", "you", "lose", "them", "to", "the", "next", "level", "if", "if", "they", "decide", "to", "recall", "the", "the", "parent", "team", "the", "Braves", "decide", "to", "call", "to", "recall", "a", "guy", "from", "triple", "A", "then", "a", "double", "A", "guy", "goes", "up", "to", "replace", "him", "and", "a", "single", "A", "guy", "goes", "up", "to", "replace", "him"],
                     "hypothesis": ["You", "lose", "the", "things", "to", "the", "following", "level", "if", "the", "people", "recall", "."],
                     "label": "entailment"}

        instance2 = {"premise": ["One", "of", "our", "number", "will", "carry", "out", "your", "instructions", "minutely", "."],
                     "hypothesis": ["A", "member", "of", "my", "team", "will", "execute", "your", "orders", "with", "immense", "precision", "."],
                     "label": "entailment"}

        assert len(instances) == 3
        def equals(fields, instance): 
        	assert [t.text for t in fields["premise"].tokens] == instance["premise"]
        	assert [t.text for t in fields["hypothesis"].tokens] == instance["hypothesis"]
        	assert fields["label"].label == instance["label"]

        equals(instances[0].fields, instance0)
        equals(instances[1].fields, instance1)
        equals(instances[2].fields, instance2)