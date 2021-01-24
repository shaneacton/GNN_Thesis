from typing import List, Tuple

from torch import Tensor
from transformers import TokenSpan

from Code.Data.Text.spacy_utils import get_entity_char_spans
from Code.HDE.Glove.glove_embedder import GloveEmbedder
from Code.HDE.edge import HDEEdge
from Code.HDE.graph import HDEGraph
from Code.HDE.graph_utils import fully_connect
from Code.HDE.node import HDENode
from Code.constants import CODOCUMENT, ENTITY


def get_glove_entities(summariser, support_embeddings, supports, glove_embedder: GloveEmbedder) \
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

        ent_summaries: List[Tensor] = []
        ent_token_spans: List[Tuple[int]] = []
        support_tokens = glove_embedder.get_words(support)
        ent_counts = {}
        for e, c_span in enumerate(ent_c_spans):
            """clips out the entities token embeddings, and summarises them"""
            try:
                entity_tokens = glove_embedder.get_words(support[c_span[0]: c_span[1]])
                if len(entity_tokens) == 0:
                    raise Exception("no entity tokens: " + repr(entity_tokens) + " char span: " + repr(c_span) + " ent: " + repr(support[c_span[0]: c_span[1]]))
                matches = find_tokens_in_token_list(support_tokens, entity_tokens)
                ent_hash = tuple(entity_tokens)
                if not ent_hash in ent_counts:
                    ent_counts[ent_hash] = 0
                if ent_counts[ent_hash] >= len(matches):
                    raise Exception("found " + repr(ent_counts[ent_hash] + 1) + " ent mentions but only " +
                                    repr(len(matches)) + " matches    ent tokens: " + repr(entity_tokens) + "   all tokens: " + repr(support_tokens))
                match = matches[ent_counts[ent_hash]]
                ent_counts[ent_hash] += 1
                ent_token_span = TokenSpan(match, match + len(entity_tokens))
            except Exception as ex:
                print("cannot get ent ", e, "token span. in supp", s, ":", support)
                print(ex)
                continue
            ent_token_spans.append(ent_token_span)
            ent_summaries.append(summariser(support_embeddings[s], ent_token_span))

        token_spans.append(ent_token_spans)
        summaries.extend(ent_summaries)

    return token_spans, summaries


def find_tokens_in_token_list(all_tokens: List[str], entity_tokens: List[str]):
    # print("ent tokens:", entity_tokens, "all:", all_tokens)
    candidates = set(range(len(all_tokens)))  # all tokens
    for w in range(len(entity_tokens)):  # loop through all entity tokens
        # iteratively eliminate all candidates which don't continue to work
        indices = [c for c in candidates if c + w < len(all_tokens) and all_tokens[c + w] == entity_tokens[w]]
        candidates = set(indices)  # all which were previously cands minus the ones which failed the next match
    if len(candidates) == 0:
        raise Exception("cannot find", entity_tokens, "in", all_tokens)
    return sorted(list(candidates))  # if the entity was mentioned multiple times, we expect multiple mathces


def add_glove_entity_nodes(graph: HDEGraph, supports, ent_token_spans: List[List[Tuple[int]]], support_tokens: List[List[str]]):
    for s, support in enumerate(supports):

        ent_spans = ent_token_spans[s]
        sup_node = graph.get_doc_nodes()[s]

        ent_node_ids = []
        tokens = support_tokens[s]
        for ent_span in ent_spans:
            text = " ".join(tokens[ent_span[0]: ent_span[1]])
            node = HDENode(ENTITY, doc_id=s, ent_token_spen=ent_span, text=text)
            ent_node_id = graph.add_node(node)
            ent_node_ids.append(ent_node_id)

            doc_edge = HDEEdge(sup_node.id_in_graph, ent_node_id, graph=graph)
            graph.add_edge(doc_edge)

        fully_connect(ent_node_ids, graph, CODOCUMENT)