from Code.GNN_Playground.Models import tokeniser, subtoken_mapper


class TokenSequence:

    """
        contains an ordered collection of word tokens and subword subtokens
        as well as the mapping between them
    """

    def __init__(self, original_text):
        self.text = original_text
        self.tokens = []  # all unsplit tokens
        self.flat_sub_tokens = []  # all split tokens
        self.sub_tokens = []  # all split tokens indexed by the parent tokens id
        subtoken_map = subtoken_mapper(original_text)
        for stm in subtoken_map:
            self.add_token_and_subtokens(stm[0], stm[1])

    def add_token_and_subtokens(self, token, subtokens):
        self.tokens.append(token)
        self.sub_tokens.append(subtokens)
        self.flat_sub_tokens.extend(subtokens)

    def find_char_seq_in_subtokens(self, start_char_id, end_char_id):
        sub_string = self.text[start_char_id, end_char_id]
        return self.find_string_in_subtokens(sub_string)

    def find_string_in_subtokens(self, string):
        sub_tokens = tokeniser(string)
        print("query sub tokens:",sub_tokens)
        print("sequence sub tokens:", self.sub_tokens)
        print("flat sub tokens:",self.flat_sub_tokens)
        sub_token_index = 0
        num_subtokens = len(self.sub_tokens)
        matches = []  # list of tuples [start,end) ids of the matching subtokens

        def does_match_from(start_id):
            for i in range(1, len(sub_tokens)):
                if sub_tokens[i] != self.flat_sub_tokens[start_id + i]:
                    return False
            return True

        while sub_token_index < num_subtokens:
            try:
                next_match_id = self.flat_sub_tokens.index(sub_tokens[0], sub_token_index, num_subtokens + 1)
                print("found start match id:",next_match_id)
                if does_match_from(next_match_id):
                    matches.append((next_match_id, next_match_id + len(sub_tokens)))
                    print("found match: ", matches[-1])
                sub_token_index = next_match_id + 1
            except:  # no more matches
                break


if __name__ == "__main__":
    text = "the apple is wick. super bad. I even thought it was stadia." * 2
    ts = TokenSequence(text)
    ts.find_string_in_subtokens("it was stadia")
