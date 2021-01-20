import re
from typing import Tuple, List

from torch import Tensor
from transformers import LongformerTokenizerFast, BatchEncoding, TokenSpan

from Code.Data.Text.spacy_utils import get_entity_char_spans
from Code.HDE.edge import HDEEdge
from Code.HDE.graph import HDEGraph
from Code.HDE.node import HDENode
from Code.constants import DOCUMENT, ENTITY, CODOCUMENT, CANDIDATE, COMENTION

only_letters = re.compile('[^a-zA-Z]')
def clean(text):
    return only_letters.sub('', text.lower())


def connect_unconnected_entities(graph: HDEGraph):
    """should be called last, after all other nodes are connected"""
    for e1, ent_node1 in enumerate(graph.get_entity_nodes()):
        for e2, ent_node2 in enumerate(graph.get_entity_nodes()):
            if e1 == e2:  # same mention
                continue

            edge = HDEEdge(ent_node1.id_in_graph, ent_node2.id_in_graph, type=ENTITY)
            graph.safe_add_edge(edge)


def connect_entity_mentions(graph: HDEGraph):

    for e1, ent_node1 in enumerate(graph.get_entity_nodes()):
        ent_text1 = clean(ent_node1.text)

        for e2, ent_node2 in enumerate(graph.get_entity_nodes()):
            if e1 == e2:  # same mention
                continue
            ent_text2 = clean(ent_node2.text)
            if ent_text1 in ent_text2 or ent_text2 in ent_text1:
                """same entity, different mention"""
                edge = HDEEdge(ent_node1.id_in_graph, ent_node2.id_in_graph, type=COMENTION)
                graph.safe_add_edge(edge)


def connect_candidates_and_entities(graph: HDEGraph):
    """type 3 conects candidates to their entity mentions"""
    for cand_node in graph.get_candidate_nodes():

        for ent_node in graph.get_entity_nodes():

            edge = HDEEdge(cand_node.id_in_graph, ent_node.id_in_graph, graph=graph)
            graph.add_edge(edge)


def add_candidate_nodes(graph: HDEGraph, candidates: List[str], supports: List[str]):
    """
        type 1 and 6.
        creates cand nodes, connects them to supports/docs which contain their mentions
        also fully connects all candidate nodes with eachother
    """
    cand_ids = []
    for c, candidate in enumerate(candidates):
        node = HDENode(CANDIDATE, candidate_id=c, text=candidate)
        c_id = graph.add_node(node)

        cand_ids.append(c_id)

        cand_node = graph.get_candidate_nodes()[c]
        clean_cand = only_letters.sub('', candidate.lower())
        for s, supp in enumerate(supports):
            supp = clean(supp)
            if supp in clean_cand or clean_cand in supp:

                edge = HDEEdge(s, cand_node.id_in_graph, graph=graph)  # type 1
                graph.add_edge(edge)

    fully_connect(cand_ids, graph, None)  # type 6


def add_doc_nodes(graph: HDEGraph, supports: List[str]):
    """document nodes should be added first so their indexes match their graph node indexes"""
    for s, support in enumerate(supports):
        node = HDENode(DOCUMENT, text=support, doc_id=s)
        graph.add_node(node)


def fully_connect(node_ids, graph, type):
    for id1 in node_ids:
        for id2 in node_ids:
            if id1 == id2:
                continue
            edge = HDEEdge(id1, id2, type=type, graph=graph)
            graph.safe_add_edge(edge)


def add_entity_nodes(graph: HDEGraph, supports, support_encodings: List[BatchEncoding],
                     ent_token_spans: List[List[Tuple[int]]], tokeniser: LongformerTokenizerFast):
    for s, support in enumerate(supports):

        ent_spans = ent_token_spans[s]
        sup_enc = support_encodings[s]
        sup_node = graph.get_doc_nodes()[s]

        ent_node_ids = []
        for ent_span in ent_spans:
            ent_tok_ids = sup_enc["input_ids"][ent_span[0]: ent_span[1]]
            text = tokeniser.decode(ent_tok_ids)
            node = HDENode(ENTITY, doc_id=s, ent_token_spen=ent_span, text=text)
            ent_node_id = graph.add_node(node)
            ent_node_ids.append(ent_node_id)

            doc_edge = HDEEdge(sup_node.id_in_graph, ent_node_id, graph=graph)
            graph.add_edge(doc_edge)

        fully_connect(ent_node_ids, graph, CODOCUMENT)


def charspan_to_tokenspan(encoding, char_span: Tuple[int]) -> TokenSpan:
    start = encoding.char_to_token(char_index=char_span[0], batch_or_char_index=0)

    recoveries = [-1, 0, -2, -3]  # which chars to try. To handle edge cases such as ending on dbl space ~ '  '
    end = None
    while end is None:
        if len(recoveries) == 0:
            raise Exception(
                "could not get end token span from char span:" + repr(char_span) + " num tokens: " + repr(
                    len(encoding.tokens())) + " ~ " + repr(encoding))

        offset = recoveries.pop(0)
        end = encoding.char_to_token(char_index=char_span[1] + offset, batch_or_char_index=0)

    span = TokenSpan(start - 1, end)  # -1 to discount the <s> token
    return span


def get_entities(summariser, support_embeddings, support_encodings, supports) \
        -> Tuple[List[List[Tuple[int]]], List[Tensor]]:
    """
        token_spans is indexed list[support_no][ent_no]
        summaries is a flat list
    """
    token_spans: List[List[Tuple[int]]] = []
    summaries: List[Tensor] = []

    for s, support in enumerate(supports):
        """get entity node embeddings"""
        ent_c_spans = get_entity_char_spans(support)
        support_encoding = support_encodings[s]

        ent_summaries: List[Tensor] = []
        ent_token_spans: List[Tuple[int]] = []
        for e, c_span in enumerate(ent_c_spans):
            """clips out the entities token embeddings, and summarises them"""
            ent_token_span = charspan_to_tokenspan(support_encoding, c_span)
            ent_token_spans.append(ent_token_span)
            ent_summaries.append(summariser(support_embeddings[s], ent_token_span))

        token_spans.append(ent_token_spans)
        summaries.extend(ent_summaries)

    return token_spans, summaries
