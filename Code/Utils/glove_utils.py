from typing import List, Tuple

from transformers import TokenSpan

from Code.Utils.spacy_utils import get_entity_char_spans, get_noun_char_spans
from Code.Embedding.glove_embedder import GloveEmbedder
from Code.HDE.Graph.edge import HDEEdge
from Code.HDE.Graph.graph import HDEGraph
from Code.Utils.graph_utils import fully_connect
from Code.HDE.Graph.node import HDENode
from Code.constants import CODOCUMENT, ENTITY
from Config.config import conf


def get_glove_entity_token_spans(example, glove_embedder: GloveEmbedder, use_nouns=False) -> List[List[Tuple[int]]]:
    """returns a 2d list indexed [supp_id][ent_in_supp]"""
    all_token_spans: List[List[Tuple[int]]] = []
    for s, support in enumerate(example.supports):
        """get entity node embeddings"""
        if conf.use_special_entities:
            """
                here we will be using 
            """
            doc_token_spans = get_special_entity_token_spans(example, support, glove_embedder)
        else:
            """
                here we are going to be using an entity detection module. this will predict charspans for entites
                which we will turn into token spans wrt the embedder.
            """
            if use_nouns:
                ent_c_spans = get_noun_char_spans(support)
            else:
                ent_c_spans = get_entity_char_spans(support)

            doc_token_spans = get_glove_entity_token_spans_from_chars(ent_c_spans, glove_embedder, support)
        all_token_spans.append(doc_token_spans)
    return all_token_spans


def get_special_entity_token_spans(example, support, glove_embedder) -> List[Tuple[int]]:
    passage_words: List[str] = glove_embedder.get_words(support)
    subject_words: List[str] = glove_embedder.get_words(example.query_subject)
    candidate_words: List[List[str]] = [glove_embedder.get_words(cand) for cand in example.candidates]
    special_words: List[List[str]] = candidate_words + [subject_words]

    token_spans = []
    for specials in special_words:
        if len(specials) <= 0:
            continue
        start = specials[0]
        indices = [i for i, x in enumerate(passage_words) if x == start]
        for i in indices:
            corr_passage_words = passage_words[i:i+len(specials)]
            if corr_passage_words == specials:
                # print("found overlap! special:", specials, "subject:", subject_words, "cands:", candidate_words)
                token_spans.append(TokenSpan(i, i+len(specials)))
    return token_spans


def get_glove_entity_token_spans_from_chars(ent_c_spans, glove_embedder, support):
    doc_token_spans: List[Tuple[int]] = []
    support_tokens = glove_embedder.get_words(support)
    ent_counts = {}
    for e, c_span in enumerate(ent_c_spans):
        """clips out the entities token embeddings, and summarises them"""
        try:
            entity_tokens = glove_embedder.get_words(support[c_span[0]: c_span[1]])
            if len(entity_tokens) == 0:
                raise NoEntityTokens(
                    "no entity tokens: " + repr(entity_tokens) + " char span: " + repr(c_span) + " ent: " + repr(
                        support[c_span[0]: c_span[1]]))
            matches = find_tokens_in_token_list(support_tokens, entity_tokens)
            ent_hash = tuple(entity_tokens)
            if not ent_hash in ent_counts:
                ent_counts[ent_hash] = 0
            if ent_counts[ent_hash] >= len(matches):
                raise Exception("found " + repr(ent_counts[ent_hash] + 1) + " ent mentions but only " +
                                repr(len(matches)) + " matches    ent tokens: " + repr(
                    entity_tokens) + "   all tokens: " + repr(support_tokens))
            match = matches[ent_counts[ent_hash]]
            ent_counts[ent_hash] += 1
            ent_token_span = TokenSpan(match, match + len(entity_tokens))
        except NoEntityTokens as ex:
            # print(ex)
            continue
        except Exception as exx:
            # print("cannot get ent ", e, "token span. in supp", s, ":", support)
            # raise exx
            continue

        doc_token_spans.append(ent_token_span)
    return doc_token_spans


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


class NoEntityTokens(Exception):
    pass