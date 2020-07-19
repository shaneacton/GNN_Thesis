from typing import List, Dict, Union

from Code.Config import config, configuration
from Code.Data.Text.Tokenisation import tokenisation_utils
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan
from Code.Data.Text.Tokenisation.spacy_utils import get_spacy_sentences, get_spacy_coreferences
from Code.Data.Text.Tokenisation.token_sequence import TokenSequence
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Data.Text.Tokenisation.tokenisation_utils import get_passages


class TokenSpanHierarchy:
    """
    built on top of a token span. is a graph of span nodes where the only edge is "contains"
    is configuration agnostic. calculates all levels of spans from ents to sents to passages
    a context graph can be constructed from this hierarchy in a configuration aware manner

    level 0 is tokens
    level 1 is words eg entities and corefs
    level 2 is sentences
    level 3 is passages
    level 4 is documents
    """

    def __init__(self, tok_seq: TokenSequence):
        self.token_sequence = tok_seq

        self._entities: List[EntitySpan] = None
        self._corefs: List[EntitySpan] = None
        self._entities_and_corefs: List[EntitySpan] = None

        self._sentences: List[DocumentExtract] = None
        self._passages: List[DocumentExtract] = None

    @staticmethod
    def match_heirarchical_span_seqs(containing_spans: List[DocumentExtract], contained_spans: List[DocumentExtract]) -> Dict[
        TokenSpan, List[TokenSpan]]:
        """
        inputs 2 sorted span sequences such as a list of sentences and entities or passages and sentences
        each list should be in order of appearance in text
        the there should be a one to many mapping of these seq
        each containing_spans span may contain many contained_spans span

        :return: dict{span_key, List[span_values]}
        """

        mapping = {}
        v_i = 0
        for key in containing_spans:  # for each containing span
            mapping[key] = []
            while v_i < len(contained_spans) and not key.contains(contained_spans[v_i]):
                # loop until the first match is found
                v_i += 1

            while v_i < len(contained_spans) and key.contains(contained_spans[v_i]):
                contained_span = contained_spans[v_i]
                mapping[key] += [contained_span]
                v_i += 1

        # if v_i != len(contained_spans):
        #     raise Exception("matching failed to place all value spans - " + str(v_i) + "/" + str(len(contained_spans)))

        return mapping

    def __getitem__(self, item) -> Union[List[DocumentExtract], TokenSequence]:
        if item == 0:
            return self.tokens
        if item == 1:
            return self.words
        if item == 2:
            return self.sentences
        if item == 3:
            return self.passages
        if item == 4:
            return [self.full_document]

    @property
    def tokens(self):
        return self.token_sequence.subtokens

    @property
    def words(self):
        entities = "entity" in config.word_nodes
        corefs = "coref" in config.word_nodes

        if entities and not corefs:
            return self.entities
        if entities and corefs:
            return self.entities_and_corefs
        raise Exception()

    @property
    def entities(self) -> List[EntitySpan]:  # ordered, non unique
        if self._entities is None:
            self._entities = tokenisation_utils.get_entities(self.token_sequence)
        return self._entities

    @property
    def corefs(self) -> Dict[EntitySpan, List[EntitySpan]]:  # ordered, non unique
        if self._corefs is None:
            self._corefs = get_spacy_coreferences(self.token_sequence, self.entities)
        return self._corefs

    @property
    def entities_and_corefs(self) -> List[EntitySpan]:
        """:returns in order entity spans"""
        if self._entities_and_corefs is None:

            self._entities_and_corefs: List[EntitySpan] = []
            # inserts corefs right after their mains resulting in an imperfect by nearly sorted array
            [self._entities_and_corefs.extend([ent] + (self.corefs[ent] if ent in self.corefs else [])) for ent in self.entities]
            # todo sort which exploits nearly sortedness
            self._entities_and_corefs = sorted(self._entities_and_corefs,
                                               key=lambda ent: 0.5 * (ent.subtoken_indexes[0] + ent.subtoken_indexes[1]))

        return self._entities_and_corefs

    @property
    def sentences(self) -> List[DocumentExtract]:
        if self._sentences is None:
            self._sentences = get_spacy_sentences(self.token_sequence)
        return self._sentences

    @property
    def full_document(self):
        return DocumentExtract(self.token_sequence, (0, len(self.token_sequence.raw_subtokens)), configuration.DOCUMENT)

    @property
    def passages(self):
        if self._passages is None:
            self._passages = get_passages(self.token_sequence)
        return self._passages