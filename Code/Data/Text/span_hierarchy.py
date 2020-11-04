from typing import Callable, Any, Optional, List, Tuple, Dict

from transformers import BatchEncoding, TokenSpan

from Code.constants import CONTEXT, QUERY, TOKEN, WORD, SENTENCE, PARAGRAPH


def contains(large: TokenSpan, small: TokenSpan):
    left = large.start <= small.start
    right = large.end >= small.end
    # print("checking contains for", large, "and", small, "--", (left and right))

    if left and right:
        return True
    # if left or right:
    #     raise Exception("overlapping spans: " + repr(large) + " and " + repr(small))
    return False


class SpanHierarchy:
    """
    temporary structure used to connect nodes in a context graph

    a tiered set of token spans whereby all spans in lower tiers are contained in the higher tier spans
    eg tokens contained in sentences, or entities contained in passages
    here token spans are with respect to a Transformers BatchEncoding's tokens

    at each tier, the spans are in order
    spans in the hierarchy can have labels, to differentiate between entities, nouns, corefs, etc
    """

    def __init__(self, text, encoding: BatchEncoding, source, batch_id=0):
        self.batch_id = batch_id
        acceptable_sources = [CONTEXT, QUERY]
        if source not in acceptable_sources:
            raise Exception("source must be one of: " + repr(acceptable_sources))
        self.source = source
        self.text = text
        self.encoding: BatchEncoding = encoding
        self.doc = None
        self.level_order = [TOKEN, WORD, SENTENCE, PARAGRAPH]  # smallest to largest
        self.levels: Dict[str, List[TokenSpan]] = {}  # maps level name (eg WORD) to spans

        self.containing_links: Dict[TokenSpan, List[TokenSpan]] = {} # maps container to all containees

    @property
    def num_tokens(self):
        return len(self.encoding["input_ids"]) - 2  # -2 for s and /s

    def convert_charspan_to_tokenspan(self, char_span: Tuple[int]) -> TokenSpan:
        start = self.encoding.char_to_token(char_index=char_span[0], batch_or_char_index=self.batch_id)
        end = self.encoding.char_to_token(char_index=char_span[1] - 1, batch_or_char_index=self.batch_id)
        span = TokenSpan(start, end + 1)  # todo confirm + 1
        return span

    def add_spans_from_chars(self, char_span_method: Callable[[str, Optional[Any]], List[Any]], level, label=None):
        """provided a method to map from text to char spans - converts to token spans, then adds and sorts"""
        if level not in self.level_order:
            raise Exception("level must be one of: " + repr(self.level_order))
        if not label:
            label = level

        chars, self.doc = char_span_method(self.text, self.doc)
        token_spans = [self.convert_charspan_to_tokenspan(c_span) for c_span in chars]
        self._add_token_spans(token_spans, level)

    def _add_token_spans(self, token_spans: List[TokenSpan], level):
        if level not in self.levels:
            self.levels[level] = token_spans  # automatically in order
            return

        """must splice new spans in order"""
        raise NotImplementedError()

    def add_tokens(self):
        """adds token spans (size 1) for each token in the encoding"""
        t_spans = [TokenSpan(t, t + 1) for t in range(self.num_tokens)]
        self._add_token_spans(t_spans, TOKEN)

    def add_full_query(self):
        """
            the full query is represented as a SENTENCE, however in case the query is detected as multiple sentences,
            this is given a special case -> so only 1 sentence node is created
        """
        self._add_token_spans([TokenSpan(0, self.num_tokens)], SENTENCE)

    def calculate_encapsulation(self):
        """
            called after all spans have been added to the hierarchy,
            calculates the information regarding where spans are contained. Eg which tokens are in which sentences
        """
        # smallest to largest
        present_levels = [lev for lev in self.level_order if lev in self.levels]
        for l, lev in enumerate(present_levels):
            if l + 1 >= len(present_levels) :
                break  # finished linking
            small_spans = self.levels[lev]
            larger_spans = self.levels[present_levels[l+1]]
            self.link_spans(small_spans, larger_spans)

    def link_spans(self, small_spans, larger_spans):
        """some small spans may not be contained, but all large spans contain"""
        s_i = 0
        s_l = len(small_spans)
        for l in larger_spans:
            """find all smol spans contained"""
            found = False
            while not found and s_i < s_l:  # loop till find first contained smol
                found = contains(l, small_spans[s_i])
                s_i += 1
            s_i -= 1
            # now s_i is the index of the first contained smolboi
            while s_i < s_l and contains(l, small_spans[s_i]):  # loop until last contained smol is added
                if l not in self.containing_links:
                    self.containing_links[l] = []
                self.containing_links[l].append(small_spans[s_i])
                s_i += 1

