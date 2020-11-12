from typing import Callable, Any, Optional, List, Tuple, Dict

from transformers import BatchEncoding, TokenSpan

from Code.Data.Graph.Nodes.structure_node import StructureNode
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.constants import CONTEXT, QUERY, TOKEN, WORD, SENTENCE, PARAGRAPH


class SpanHierarchy:
    """
    temporary structure used to connect nodes from a single source

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
        self.levels: Dict[str, List[SpanNode]] = {}  # maps level name (eg WORD) to spans

        self.containing_links: Dict[SpanNode, List[SpanNode]] = {} # maps container to all containees

    @property
    def num_tokens(self):
        return len(self.encoding["input_ids"]) - 2  # -2 for s and /s

    @property
    def present_levels(self):
        return [lev for lev in self.level_order if lev in self.levels]

    def convert_charspan_to_tokenspan(self, char_span: Tuple[int]) -> TokenSpan:
        start = self.encoding.char_to_token(char_index=char_span[0], batch_or_char_index=self.batch_id)
        end = self.encoding.char_to_token(char_index=char_span[1] - 1, batch_or_char_index=self.batch_id)
        if end is None:
            """
                char span has landed on  a space.
                this was likely a failure of the sentence extraction system
                correct this by erring on the side of caution and taking an extra token instead of one fewer
            """
            end = self.encoding.char_to_token(char_index=char_span[1], batch_or_char_index=self.batch_id)
            # raise Exception("could not get end token span from char span:" + repr(char_span) + " num tokens: " + repr(len(self.encoding.tokens())) + " ~ " + repr(self.encoding))
        span = TokenSpan(start, end + 1)  # todo confirm + 1
        return span

    def add_spans_from_chars(self, char_span_method: Callable[[str, Optional[Any]], List[Any]], level,
                             node_type, subtype=None):
        """provided a method to map from text to char spans - converts to token spans, then adds and sorts"""
        if level not in self.level_order:
            raise Exception("level must be one of: " + repr(self.level_order))

        chars, self.doc = char_span_method(self.text, self.doc)
        try:
            token_spans = [self.convert_charspan_to_tokenspan(c_span) for c_span in chars]
        except Exception as e:
            print(e)
            raise Exception("could not add span nodes for level: " + repr(level) + " using " + repr(char_span_method))
        nodes = [node_type(s, source=self.source, subtype=subtype) for s in token_spans]
        self._add_span_nodes(nodes, level)

    def _add_span_nodes(self, token_spans: List[SpanNode], level):
        if level not in self.levels:
            self.levels[level] = token_spans  # automatically in order
            return

        """must splice new spans in order"""
        raise NotImplementedError()

    def add_tokens(self):
        """adds token spans (size 1) for each token in the encoding"""
        t_spans = [TokenNode(TokenSpan(t, t + 1), source=self.source) for t in range(self.num_tokens)]
        self._add_span_nodes(t_spans, TOKEN)

    def add_full_query(self):
        """
            the full query is represented as a SENTENCE, however in case the query is detected as multiple sentences,
            this is given a special case -> so only 1 sentence node is created
        """
        spans = [StructureNode(TokenSpan(0, self.num_tokens), source=self.source, subtype=SENTENCE)]
        self._add_span_nodes(spans, SENTENCE)

    def calculate_encapsulation(self):
        """
            called after all spans have been added to the hierarchy,
            calculates the information regarding where spans are contained. Eg which tokens are in which sentences
        """
        # smallest to largest
        for l, lev in enumerate(self.present_levels):
            if l + 1 >= len(self.present_levels) :
                break  # finished linking
            small_spans = self.levels[lev]
            larger_spans = self.levels[self.present_levels[l+1]]
            self.link_spans(small_spans, larger_spans)

    def link_spans(self, small_spans, larger_spans):
        """some small spans may not be contained, but all large spans contain"""
        s_i = 0
        s_l = len(small_spans)
        for l in larger_spans:
            """find all smol spans contained"""
            found = False
            while not found and s_i < s_l:  # loop till find first contained smol
                found = small_spans[s_i] in l
                s_i += 1
            s_i -= 1
            # now s_i is the index of the first contained smolboi
            while s_i < s_l and small_spans[s_i] in l:  # loop until last contained smol is added
                if l not in self.containing_links:
                    self.containing_links[l] = []
                self.containing_links[l].append(small_spans[s_i])
                s_i += 1

