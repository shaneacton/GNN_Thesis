from Code.Data.Text.Tokenisation.tokenisation_utils import find_seq_in_seq
from Code.Models import subtoken_mapper, basic_tokeniser


class TokenSequence:

    """
        contains an ordered collection of word tokens and subword subtokens
        as well as the mapping between them
    """

    def __init__(self, text_obj):
        self.text_obj = text_obj
        self.raw_tokens = []  # all unsplit tokens
        self.raw_subtokens = []  # all split tokens

        self.token_subtoken_map = []  # value at index i is [start,end) subtoken id's for token i
        subtoken_map = subtoken_mapper(text_obj.raw_text)
        for stm in subtoken_map:
            self.add_token_and_subtokens(stm[0], stm[1])

    def get_text_from_span(self, span):
        return " ".join(self.raw_tokens[span[0]:span[1]])

    def add_token_and_subtokens(self, token: str, subtokens: str):
        self.raw_tokens.append(token)
        num_existing_subtokens = len(self.raw_subtokens)
        self.token_subtoken_map.append((num_existing_subtokens, num_existing_subtokens + len(subtokens)))
        self.raw_subtokens.extend(subtokens)

    def convert_token_span_to_subtoken_span(self, token_span):
        t_start, t_end = token_span
        sub_start = self.token_subtoken_map[t_start][0]
        sub_end = self.token_subtoken_map[t_end-1][1] # todo test ensure -1 is correct
        return (sub_start, sub_end)

    def get_token_span_from_chars(self, start_char_id, end_char_id, subtokens=False):
        sub_string = self.text_obj.raw_text[start_char_id: end_char_id]
        matches = find_seq_in_seq(basic_tokeniser(self.text_obj.raw_text[:end_char_id]), basic_tokeniser(sub_string))
        # since the text was sheered at the end_char_id, final match must be accurate
        if not matches:
            raise Exception("Possible substring does not align with chars")
        if subtokens:
            return self.convert_token_span_to_subtoken_span(matches[-1])
        return matches[-1]

    def __eq__(self, other):
        return self.text_obj == other.text_obj

    def __hash__(self):
        return self.text_obj.__hash__()


