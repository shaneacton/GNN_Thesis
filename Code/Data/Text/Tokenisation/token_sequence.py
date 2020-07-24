from typing import List

from Code.Data.Text.Tokenisation.token import Token
from Code.Data.Text.Tokenisation.tokenisation_utils import find_seq_in_seq
from Code.Models import subtoken_mapper, basic_tokeniser


class TokenSequence:

    """
        contains an ordered collection of word tokens and subword subtokens
        as well as the mapping between them
    """

    def __init__(self, text_obj):
        self.text_obj = text_obj
        self.raw_word_tokens: List[str] = []  # all unsplit tokens
        self.raw_subtokens: List[str] = []  # all split tokens
        self._subtoken_objects: List[Token] = []

        self.token_subtoken_map = []  # value at index i is [start,end) subtoken id's for token i
        self.subtoken_token_map = {}  # stm[sub_id]=containing token_id

        subtoken_map = subtoken_mapper(text_obj.raw_text)
        for stm in subtoken_map:
            self.add_word_token_and_subtokens(stm[0], stm[1])

    @property
    def subtokens(self):
        if not self._subtoken_objects:
            self._subtoken_objects = self.get_subtoken_objects()
        return self._subtoken_objects

    def get_subtoken_objects(self):
        subs = []
        for i in range(len(self.raw_subtokens)):
            obj = Token(self, (i,i+1))
            subs.append(obj)
        return subs

    def get_text_from_word_span(self, span):
        return " ".join(self.raw_word_tokens[span[0]:span[1]])

    def get_text_from_subspan(self, span):
        return " ".join(self.raw_subtokens[span[0]:span[1]])

    def add_word_token_and_subtokens(self, token: str, subtokens: str):
        num_preexisting_subtokens = len(self.raw_subtokens)
        num_preexisting_tokens = len(self.raw_word_tokens)
        self.raw_word_tokens.append(token)
        self.raw_subtokens.extend(subtokens)

        num_new_subtokens = len(subtokens)

        for i in range(num_new_subtokens):
            # assign all new subtokens as being contained by the token
            self.subtoken_token_map[num_preexisting_subtokens + i] = num_preexisting_tokens
        self.token_subtoken_map.append((num_preexisting_subtokens, num_preexisting_subtokens + num_new_subtokens))

    def convert_word_token_span_to_subtoken_span(self, token_span):
        t_start, t_end = token_span
        sub_start = self.token_subtoken_map[t_start][0]
        sub_end = self.token_subtoken_map[t_end-1][1] # todo test ensure -1 is correct
        return sub_start, sub_end

    def convert_subtoken_span_to_word_token_span(self, subtoken_span):
        s_start, s_end = subtoken_span
        t_start = self.subtoken_token_map[s_start]
        t_end = self.subtoken_token_map[s_end-1]
        return t_start, t_end+1

    def get_word_token_span_from_chars(self, start_char_id, end_char_id, subtokens=False):
        """
        returns the subtoken span if subs=true
        """
        sub_string = self.text_obj.raw_text[start_char_id: end_char_id]
        matches = find_seq_in_seq(basic_tokeniser(self.text_obj.raw_text[:end_char_id]), basic_tokeniser(sub_string))
        # since the text was sheered at the end_char_id, final match must be accurate
        if not matches:
            raise Exception("Possible substring does not align with chars")
        if subtokens:
            return self.convert_word_token_span_to_subtoken_span(matches[-1])
        return matches[-1]

    def __eq__(self, other):
        return self.text_obj == other.text_obj

    def __hash__(self):
        return self.text_obj.__hash__()

    def __len__(self):
        return len(self.raw_subtokens)




