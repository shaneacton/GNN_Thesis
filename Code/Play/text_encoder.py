from typing import Dict

from transformers.tokenization_utils_base import TruncationStrategy

from Code.Data.Text.text_utils import words, question


class TextEncoder:

    def __init__(self, tokeniser):
        self.tokeniser = tokeniser

    def get_basic_features(self, example: Dict):
        # print("example before:", example)
        encodings = self.tokeniser(question(example), example['context'], pad_to_max_length=True, max_length=512,
                                   truncation=True)
        start_positions, end_positions = self.get_answer_token_span(example, encodings)
        example.pop("answers", None)
        example["start_positions"] = start_positions
        example["end_positions"] = end_positions
        # print("example after:", example)
        return example

    def get_qa_features(self, example: Dict):
        """ready for a Longformer"""
        print("type:", type(example))
        if 'candidates' in example:
            encoding = self._get_candidate_features(example)
        else:
            encoding = self._get_span_features(example)
        print("ex:", example)
        print("encode:", encoding)
        print("tokens:", encoding.tokens())
        print("word ids:", encoding.words())
        print("words:", words(encoding, question(example), example['context']))
        return encoding

    def _get_candidate_features(self, example):
        pass

    def get_answer_token_span(self, example, qa_encoding, context_encoding=None):
        """
            here we will compute the start and end position of the answer in the whole example
            as the example is encoded like this <s> question</s></s> context</s>
            and we know the postion of the answer in the context
            we can just find models the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
            this will give us the position of the answer span in whole example
        """
        if context_encoding is None:
            context_encoding = self.tokeniser.encode_plus(example['context'])

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        start_idx, end_idx = self.get_correct_span_alignement(example['context'], example['answers'])
        start_positions_context = context_encoding.char_to_token(start_idx)
        end_positions_context = context_encoding.char_to_token(end_idx - 1)

        sep_idx = qa_encoding['input_ids'].index(self.tokeniser.sep_token_id)  # len(query_tokens)
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1
        return start_positions, end_positions

    def _get_span_features(self, example):
        # the example is encoded like this <s> question</s></s> context</s>

        encodings = self.tokeniser(question(example), example['context'], pad_to_max_length=True, max_length=512,
                                   truncation=True)
        start_positions, end_positions = self.get_answer_token_span(example, encodings)

        if end_positions > 512:
            start_positions, end_positions = 0, 0

        encodings.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encodings['attention_mask']})
        return encodings

    @staticmethod
    def get_correct_span_alignement(context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        gold_text = answer['text'][0]
        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()
