from Code.Data.Text.Tokenisation.token_span import TokenSpan


class Token(TokenSpan):

    def __init__(self, token_sequence, subtoken_indexes):
        super().__init__(token_sequence, subtoken_indexes)
