from spacy.language import Language
from spacy.tokens.doc import Doc

from Code.Data.Tokenisation.entity import Entity
from Code.Models import tokeniser, subtoken_mapper, basic_tokeniser

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()


class TokenSequence:

    """
        contains an ordered collection of word tokens and subword subtokens
        as well as the mapping between them
    """

    def __init__(self, original_text):
        self.text = original_text
        self.tokens = []  # all unsplit tokens
        self.sub_tokens = []  # all split tokens
        self.token_subtoken_map = []  # value at index i is (start,end) subtoken id's for token i
        subtoken_map = subtoken_mapper(original_text)
        for stm in subtoken_map:
            self.add_token_and_subtokens(stm[0], stm[1])

        self._entities = None

    @property
    def entities(self):
        if self._entities is None:
            self._entities = self.get_entities()
        return self._entities

    def add_token_and_subtokens(self, token, subtokens):
        self.tokens.append(token)
        num_existing_subtokens = len(self.sub_tokens)
        self.token_subtoken_map.append((num_existing_subtokens, num_existing_subtokens + len(subtokens) -1))
        self.sub_tokens.extend(subtokens)

    def convert_token_span_to_subtoken_span(self, token_span):
        t_start, t_end = token_span
        sub_start = self.token_subtoken_map[t_start][0]
        sub_end = self.token_subtoken_map[t_end][1]
        return (sub_start, sub_end)

    def find_char_seq_in_subtokens(self, start_char_id, end_char_id):
        sub_string = self.text[start_char_id, end_char_id]
        return self.find_string_in_subtokens(sub_string)

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

    def get_entities(self):
        """
            uses the unprocessed text to perform NER using Spacy.
        """
        processed: Doc = nlp(self.text)

        unique_entity_names = set([ent.text for ent in processed.ents])
        entity_matches = {ent: self.find_string_in_tokens(ent) for ent in unique_entity_names}

        entities = []

        for ent in processed.ents:
            ent_text = ent.text
            next_match = entity_matches[ent_text][0]

            entity_matches[ent_text] = entity_matches[ent_text][1:]
            entity = Entity(ent, next_match)
            entities.append(entity)

        return entities

if __name__ == "__main__":
    text = """European authorities fined Google Stadia a record $5.1 billion on Wednesday for abusing its power in the 
    mobile phone market and ordered the company to alter its practices """ * 2

    ts = TokenSequence(text)

    print(ts.entities)
    spans = [ent.token_span for ent in ts.entities]
    sub_spans = [ts.convert_token_span_to_subtoken_span(span) for span in spans]
    print(sub_spans)
    subs = [ts.sub_tokens[ss[0]:ss[1]] for ss in sub_spans]
    print(subs)