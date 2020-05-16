from typing import List, Dict, Tuple

from neuralcoref.neuralcoref import Cluster
from spacy.tokens.doc import Doc
import neuralcoref

from Code.Data.Text.Tokenisation.entity import Entity
from Code.Models import tokeniser, subtoken_mapper, basic_tokeniser

import en_core_web_sm
nlp = en_core_web_sm.load()
neuralcoref.add_to_pipe(nlp)


class TokenSequence:

    """
        contains an ordered collection of word tokens and subword subtokens
        as well as the mapping between them
    """

    def __init__(self, text):
        self.text = text
        self.tokens = []  # all unsplit tokens
        self.sub_tokens = []  # all split tokens
        self.token_subtoken_map = []  # value at index i is [start,end) subtoken id's for token i
        subtoken_map = subtoken_mapper(text.text)
        for stm in subtoken_map:
            self.add_token_and_subtokens(stm[0], stm[1])

        self._entities = None
        self._corefs = None
        self._spacy_processed_doc = None

    @property
    def spacy_processed_doc(self):
        if self._spacy_processed_doc is None:
            self._spacy_processed_doc = nlp(self.text.text)
        return self._spacy_processed_doc

    @property
    def entities(self) -> List[Entity]:  # ordered, non unique
        if self._entities is None:
            self._entities = self.get_entities(spacy_processed_doc=self.spacy_processed_doc)
        return self._entities

    @property
    def corefs(self) -> Dict[Entity, List[Entity]]:  # ordered, non unique
        if self._corefs is None:
            self._corefs = self.get_coreferences(spacy_processed_doc=self.spacy_processed_doc)
        return self._corefs

    def add_token_and_subtokens(self, token, subtokens):
        self.tokens.append(token)
        num_existing_subtokens = len(self.sub_tokens)
        self.token_subtoken_map.append((num_existing_subtokens, num_existing_subtokens + len(subtokens)))
        self.sub_tokens.extend(subtokens)

    def convert_token_span_to_subtoken_span(self, token_span):
        t_start, t_end = token_span
        sub_start = self.token_subtoken_map[t_start][0]
        sub_end = self.token_subtoken_map[t_end-1][1] # todo test ensure -1 is correct
        return (sub_start, sub_end)

    def get_token_span_from_chars(self, start_char_id, end_char_id, subtokens=False):
        sub_string = self.text.text[start_char_id: end_char_id]
        matches = self.find_seq_in_seq(basic_tokeniser(self.text.text[:end_char_id]), basic_tokeniser(sub_string))
        # since the text was sheered at the end_char_id, final match must be accurate
        if not matches:
            raise Exception("Possible substring does not align with chars")
        if subtokens:
            return self.convert_token_span_to_subtoken_span(matches[-1])
        return matches[-1]

    @staticmethod
    def find_seq_in_seq(seq, query):
        seq_index = 0
        num_seq_tokens = len(seq)
        num_query_tokens = len(query)
        matches = []  # list of tuples [start,end) ids of the matching subtokens

        def does_match_from(start_id):
            for i in range(1, num_query_tokens):
                if query[i] != seq[start_id + i]:
                    return False
            return True

        while seq_index < num_seq_tokens:
            try:
                next_match_id = seq.index(query[0], seq_index, num_seq_tokens + 1)
                if does_match_from(next_match_id):
                    matches.append((next_match_id, next_match_id + num_query_tokens))
                seq_index = next_match_id + 1
            except:  # no more matches
                break

        return matches

    def find_string_in_tokens(self, string):
        return self.find_seq_in_seq(self.tokens, basic_tokeniser(string))

    def find_string_in_subtokens(self, string):
        return self.find_seq_in_seq(self.sub_tokens, tokeniser(string))

    def get_entities(self, spacy_processed_doc=None):
        """
        uses the unprocessed text to perform NER using Spacy.
        """
        processed: Doc = nlp(self.text.text) if not spacy_processed_doc else spacy_processed_doc
        entities = []

        for ent in processed.ents:
            exact_match = self.get_token_span_from_chars(ent.start_char, ent.end_char)
            entity = Entity(ent, self, exact_match)
            entities.append(entity)

        return entities

    def get_coreferences(self, spacy_processed_doc=None):
        processed: Doc = nlp(self.text.text) if not spacy_processed_doc else spacy_processed_doc
        clusters: List[Cluster] = processed._.coref_clusters

        corefs = {}
        for cluster in clusters:
            token_matches = [self.get_token_span_from_chars(mention.start_char, mention.end_char)
                             for mention in cluster.mentions]
            entities = [Entity(cluster.mentions[i], self, token_matches[i], is_coref=i>0)
                        for i in range(len(token_matches))]
            corefs[entities[0]] = entities[1:]  # ent[0] is main, rest are corefs

        corefs = {ent: corefs[ent] for ent in ts.entities if ent in corefs}
        # replaces the main ents produced by the ents produced by NER as these have more metadata
        return corefs

    def __eq__(self, other):
        return self.text == other.text

    def __hash__(self):
        return self.text.__hash__()


if __name__ == "__main__":
    text = """European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the 
    mobile phone market and ordered the company to alter its practices. """ * 1

    from Code.Data.Text.text import Text
    ts = TokenSequence(Text(text))

    print("entities:",ts.entities)
    [ent.get_embedding() for ent in ts.entities]
    spans = [ent.token_span for ent in ts.entities]
    sub_spans = [ts.convert_token_span_to_subtoken_span(span) for span in spans]
    subs = [ts.sub_tokens[ss[0]:ss[1]] for ss in sub_spans]
    print("ent sub toks:", subs)
    print("corefs:", ts.corefs)
