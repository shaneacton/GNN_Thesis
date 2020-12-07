from typing import Dict, List, Tuple

from transformers import LongformerTokenizerFast

from Code.Data.Text.text_utils import words, question, context, is_batched, candidates, context_key


class TextEncoder:

    def __init__(self, tokeniser: LongformerTokenizerFast):
        self.tokeniser: LongformerTokenizerFast = tokeniser

    def get_processed_example(self, example: Dict):
        """
            pytorch requires lists to  be same size inside a dataloader batch
            fields with variable length arrays must be reduced to single elements
        """
        if is_batched(example):
            raise Exception("cannot get features from batched example " + repr(example))
        if "candidates" in example:
            return self.get_processed_candidate_example(example)
        return self.get_processed_span_example(example)

    def get_processed_span_example(self, example: Dict):
        """replaces answers list with a start,end position token id of the first answer"""
        # print("example before:", example)
        encodings = self.get_qa_encoding(example)
        start_positions, end_positions = self.get_answer_token_span(example, encodings)
        example.pop("answers", None)
        example["start_positions"] = start_positions
        example["end_positions"] = end_positions
        example["context"] = context(example).strip()
        if end_positions > len(self.get_context_encoding(example)["input_ids"]):
            # todo remove check, or only do once
            raise Exception("span answer = " + repr((start_positions, end_positions)) + " but only "
                            + repr(len(self.get_context_encoding(example)["input_ids"])) + " context tokens")
        # print("example after:", example)
        return example

    def get_processed_candidate_example(self, example: Dict):
        """
            replaces the candidates list with a single candidates string via concat
            replaces answer with a cand id
            replaces supports list with single context string
        """
        # print("cand ex:", candidates(example), question(example))
        # raise Exception()
        cands: List[str] = candidates(example)
        answer = example["answer"]
        correct_cand_id = cands.index(answer)
        example["candidates"] = self.get_cands_string(cands)
        example["answer"] = correct_cand_id
        ctx = context(example)
        example.pop(context_key(example))  # remove supports field
        example["context"] = ctx.strip()
        # print("process cands ex:", example)
        return example

    def get_encoding(self, example: Dict):
        """
            returns <context><query>[all candidates]
            checks if candidates are present.
        """
        if is_batched(example):
            raise Exception("cannot get encoding from batched example " + repr(example))
        # print("getting encoding for ex: " + repr(example) + "\n has cands:", ("candidates" in example))
        if "candidates" in example:
            return self.get_mcqa_encoding(example)
        return self.get_qa_encoding(example)

    def get_context_encoding(self, example):
        if is_batched(example):
            cs = context(example)
            return [self.tokeniser.encode_plus(c) for c in cs]
        return self.tokeniser.encode_plus(context(example))

    def get_question_encoding(self, example):
        if is_batched(example):
            qs = question(example)
            return [self.tokeniser.encode_plus(q) for q in qs]
        return self.tokeniser.encode_plus(question(example))

    def get_candidates_encoding(self, example):
        if is_batched(example):
            cands = candidates(example)
            return [self.tokeniser.encode_plus(self.get_cands_string(c)) for c in cands]
        return self.tokeniser.encode_plus(self.get_cands_string(candidates(example)))

    def get_cands_string(self, cands: List[str]):
        if isinstance(cands, str):
            if self.tokeniser.sep_token not in cands:
                raise Exception("cannot pass single candidate: " + repr(cands))
            # print("Tex enc: asking for cands string, but provided cands string")
            return cands
        return self.tokeniser.sep_token.join(cands)

    def get_mcqa_encoding(self, example: Dict):
        """concats <context><query><all candidates> and encodes as <context> , <query><all candidates>"""
        cands = candidates(example)
        cands_str = self.get_cands_string(cands)
        q_cans = question(example) + self.tokeniser.sep_token + cands_str
        # print("q_cans:", q_cans)
        encoding = self.tokeniser(context(example), q_cans)
        # print("encoding:", encoding)

        return encoding

    def get_qa_encoding(self, example: Dict, pad=False):
        """
            concats <context><query>
            this is opposed to the Transformers pattern which uses <query><context>
            before_sep_token=False should be used when calculating global att mask
        """
        if pad:
            return self.tokeniser(example['context'], question(example), pad_to_max_length=True, max_length=512,
                                   truncation=True)
        return self.tokeniser(example['context'], question(example))

    def get_longformer_qa_features(self, example: Dict):
        """ready for a Longformer"""
        if 'candidates' in example:
            encoding = self._get_longformer_candidate_features(example)
        else:
            encoding = self._get_longformer_span_features(example)
        # print("ex:", example)
        # print("encode:", encoding)
        # print("tokens:", encoding.tokens())
        # print("word ids:", encoding.words())
        # print("words:", words(encoding, question(example), context(example)))
        return encoding

    def _get_longformer_candidate_features(self, example):
        """encoded like <context><query><all candidates>"""
        encoding = self.get_mcqa_encoding(example)
        cands = candidates(example)
        # print("example:", example, "\ncands:", cands)
        answer = example["answer"]
        ans_id = cands.index(answer)
        start_positions, end_positions = self.get_answer_candidate_token_span(example, encoding, ans_id)

        encoding.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encoding['attention_mask'], "answer": ans_id})
        # print("ans id:", ans_id, "enc:", encoding)
        return encoding

    def get_answer_candidate_token_span(self, example, mcqa_encoding, answer_id) -> Tuple[int, int]:
        """
            mcqa is treated as span selection where the text is <context><query><all candidates>
            :returns the span corr to the correct candidates tokens in the full encoding
        """
        cands = candidates(example)
        cands_str = self.get_cands_string(cands)
        cands_encoding = self.tokeniser(cands_str)
        sep_id = self.tokeniser.sep_token_id
        starts = [0]
        ends = []
        # print("cands ids:", type(cands_encoding["input_ids"]), cands_encoding["input_ids"])
        for i, tok_id in enumerate(cands_encoding["input_ids"]):
            if tok_id == sep_id:
                if ends:  # starts of cand spans
                    starts.append(ends[-1])
                ends.append(i)  # ends of cand spans

        # print("cands enc:", cands_encoding)
        # print("cands spans:", [(starts[i], ends[i]) for i in range(len(starts))])
        # print("ans cand span:", (starts[answer_id], ends[answer_id]))

        con_query_len = len(mcqa_encoding["input_ids"]) - len(cands_encoding["input_ids"])
        # print("full mcqa:", len(mcqa_encoding["input_ids"]), "cands:", len(cands_encoding["input_ids"]), "con quer:", con_query_len)
        # print("full toks:", mcqa_encoding.tokens())

        span_in_full_enc = (starts[answer_id] + con_query_len, ends[answer_id] + con_query_len)
        # print("span in full enc:", span_in_full_enc)

        return span_in_full_enc

    def _get_longformer_span_features(self, example):
        """ready for a longformer which predicts a span. ie: LongformerForQuestionAnswering"""
        # the example is encoded like this <s> question</s></s> context</s>

        encoding = self.get_encoding(example)
        start_positions, end_positions = self.get_answer_token_span(example, encoding)

        if end_positions > 512:
            start_positions, end_positions = 0, 0

        encoding.update({'start_positions': start_positions,
                          'end_positions': end_positions,
                          'attention_mask': encoding['attention_mask']})
        return encoding

    def get_answer_token_span(self, example, qa_encoding) -> Tuple[int, int]:
        """
            example is encoded like this <s> context</s></s> question</s>
            find out which tokens the start,end chars belong to
            :returns val from 0->num_ctx_tokens
        """

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text
        start_idx, end_idx = self.get_correct_span_alignement(context(example), example['answers'])
        start_positions_context = qa_encoding.char_to_token(start_idx)
        end_positions_context = qa_encoding.char_to_token(end_idx - 1)

        return start_positions_context, end_positions_context

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
