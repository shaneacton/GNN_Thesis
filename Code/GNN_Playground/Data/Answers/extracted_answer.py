from Code.GNN_Playground.Data.Answers.answers import Answer
from Code.GNN_Playground.Models import bert_tokeniser, indexer


class ExtractedAnswer(Answer):
    """
        an extracted answer is a direct rip from the context
        it can be expressed as a start and end char index pair
    """

    def __init__(self, text, start_char_id, tokens):
        super().__init__(text)
        self.start_char_id = start_char_id
        self.end_char_id = start_char_id + len(text)

        self.start_token_id = None
        self.end_token_id = None

        print("extracted answer: ", text,"\ntokens:",tokens)
        print(type(tokens[0]))

    def __repr__(self):
        return self.text + "\t\t- char ids : ["+repr(self.start_char_id)+", " + repr(self.end_char_id) + "]"
