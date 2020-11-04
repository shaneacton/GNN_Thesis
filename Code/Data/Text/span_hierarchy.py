from typing import Callable, Any, Optional, List, Tuple, Dict

from transformers import BatchEncoding, TokenSpan

from Code.constants import CONTEXT, QUERY, TOKEN, WORD, SENTENCE, PARAGRAPH


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
        self.level_order = [TOKEN, WORD, SENTENCE, PARAGRAPH]
        self.levels: Dict[str, List[TokenSpan]] = {}  # maps level name (eg WORD) to spans

    def convert_charspan_to_tokenspan(self, char_span: Tuple[int]) -> TokenSpan:
        start = self.encoding.char_to_token(char_index=char_span[0], batch_or_char_index=self.batch_id)
        end = self.encoding.char_to_token(char_index=char_span[1], batch_or_char_index=self.batch_id)
        span = TokenSpan(start, end)
        return span

    def add_spans_from_chars(self, char_span_method: Callable[[str, Optional[Any]], List[Any]], level, label=None):
        """provided a method to map from text to char spans - converts to token spans, then adds and sorts"""
        if level not in self.level_order:
            raise Exception("level must be one of: " + repr(self.level_order))
        if not label:
            label = level

        chars, self.doc = char_span_method(self.text, self.doc)
        token_spans = [self.convert_charspan_to_tokenspan(c_span) for c_span in chars]
        self.add_token_spans(token_spans, level)

    def add_token_spans(self, token_spans: List[TokenSpan], level):
        if level not in self.levels:
            self.levels[level] = token_spans  # automatically in order
            return

        """must splice new spans in order"""
        raise NotImplementedError()
