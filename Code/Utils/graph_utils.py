from __future__ import annotations

import re
import zlib
from typing import TYPE_CHECKING
from typing import Tuple, List

from torch import Tensor
from transformers import LongformerTokenizerFast, BatchEncoding, TokenSpan

from Code.Embedding.glove_embedder import GloveEmbedder
from Code.HDE.Graph.edge import HDEEdge
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.node import HDENode
from Code.Utils.spacy_utils import get_entity_char_spans, get_sentence_char_spans
from Code.constants import DOCUMENT, ENTITY, CANDIDATE, COMENTION, SENTENCE
from Config.config import conf
from Viz.graph_visualiser import get_file_path, render_graph2

if TYPE_CHECKING:
    from Code.Training.wikipoint import Wikipoint

_regex = None


def get_regex():
    global _regex
    if _regex is None:
        _regex = re.compile('[^a-zA-Z0-9]')
        return _regex

    return _regex


def clean(text):
    cleaner = get_regex().sub('', text.lower())

    """
        if we are using strict matching, then words preceding words like 'the' can cause a misamtch.
        this is because entity detection is imperfect
    """
    words = cleaner.split(" ")
    removal_words = {"the", "a", "of"}
    safe_words = []
    for w in words:
        if w not in removal_words:
            safe_words.append(w)
    cleaner = " ".join(safe_words)
    return cleaner


def connect_unconnected_entities(graph: HDEGraph):
    """
        type 7. should be called last, after all other nodes are connected
    """

    for e1, ent_node1 in enumerate(graph.get_entity_nodes()):
        for e2, ent_node2 in enumerate(graph.get_entity_nodes()):
            if e1 == e2:  # same mention
                continue
            if graph.has_connection(ent_node1.id_in_graph, ent_node2.id_in_graph):
                continue
            edge = HDEEdge(ent_node1.id_in_graph, ent_node2.id_in_graph, type=ENTITY)
            graph.add_edge(edge)


def similar(text1, text2):
    return text1.lower().strip() == text2.lower().strip()


def connect_entity_mentions(graph: HDEGraph, all_cases=True):
    """
        type 5. mentions of the same entity.
        currently connects all such cases, not only for query and cand ents as in HDE
    """
    # if not all_cases:

    for e1, ent_node1 in enumerate(graph.get_entity_nodes()):
        ent_text1 = clean(ent_node1.text)

        for e2, ent_node2 in enumerate(graph.get_entity_nodes()):
            if e1 == e2:  # same mention
                continue

            ent_text2 = clean(ent_node2.text)
            if similar(ent_text1, ent_text2):
                """same entity, different mention"""
                if graph.has_connection(ent_node1.id_in_graph, ent_node2.id_in_graph):
                    continue

                edge = HDEEdge(ent_node1.id_in_graph, ent_node2.id_in_graph, type=COMENTION)
                graph.add_edge(edge)


def connect_candidates_and_entities(graph: HDEGraph):
    """type 3 conects candidates to their entity mentions"""
    for cand_node in graph.get_candidate_nodes():

        for ent_node in graph.get_entity_nodes():
            if similar(cand_node.text, ent_node.text):
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
        clean_cand = clean(candidate)
        for s, supp in enumerate(supports):
            supp = clean(supp)
            if clean_cand in supp:
                edge = HDEEdge(s, cand_node.id_in_graph, graph=graph)  # type 1
                graph.add_edge(edge)

    fully_connect(cand_ids, graph, None)  # type 6


def add_doc_nodes(graph: HDEGraph, supports: List[str]):
    """document nodes should be added first so their indexes match their graph node indexes"""
    for s, support in enumerate(supports):
        node = HDENode(DOCUMENT, text=support, doc_id=s)
        graph.add_node(node)


def get_entity_nodes(supports, ent_token_spans: List[List[Tuple[int]]],
                     tokeniser: LongformerTokenizerFast=None, support_encodings: List[BatchEncoding]=None,
                     glove_embedder: GloveEmbedder=None,
                     get_sentence_spans=False):

    all_nodes = []
    for s, support in enumerate(supports):

        ent_spans = ent_token_spans[s]
        nodes = []
        for ent_span in ent_spans:
            if glove_embedder is not None:
                toks = glove_embedder.get_words(support)
                text = " ".join(toks[ent_span[0]: ent_span[1]])
            else:
                ent_tok_ids = support_encodings[s]["input_ids"][ent_span[0]: ent_span[1]]
                text = tokeniser.decode(ent_tok_ids)
            node_type = ENTITY if not get_sentence_spans else SENTENCE
            node = HDENode(node_type, doc_id=s, ent_token_spen=ent_span, text=text)
            nodes.append(node)
        all_nodes.append(nodes)
    return all_nodes


def add_entity_nodes(graph: HDEGraph, supports, ent_token_spans: List[List[Tuple[int]]],
                     tokeniser: LongformerTokenizerFast=None, support_encodings: List[BatchEncoding]=None,
                     glove_embedder: GloveEmbedder=None):

    all_nodes = get_entity_nodes(supports, ent_token_spans, tokeniser, support_encodings, glove_embedder)

    for s, support in enumerate(supports):

        ent_spans = ent_token_spans[s]
        sup_node = graph.get_doc_nodes()[s]

        nodes = all_nodes[s]
        for i, ent_span in enumerate(ent_spans):
            node = nodes[i]
            ent_node_id = graph.add_node(node)

            doc_edge = HDEEdge(sup_node.id_in_graph, ent_node_id, graph=graph)
            graph.add_edge(doc_edge)


def charspan_to_tokenspan(encoding: BatchEncoding, char_span: Tuple[int]) -> TokenSpan:
    start = encoding.char_to_token(batch_or_char_index=char_span[0])
    if start is None:
        start = encoding.char_to_token(batch_or_char_index=char_span[0]+1)
    if start is None:
        raise Exception("cannot get token span from charspan:", char_span, "given:", encoding.tokens())

    recoveries = [-1, 0, -2, -3]  # which chars to try. To handle edge cases such as ending on dbl space ~ '  '
    end = None
    while end is None:
        if len(recoveries) == 0:
            raise Exception(
                "could not get end token span from char span:" + repr(char_span) + " num tokens: " + repr(
                    len(encoding.tokens())) + " ~ " + repr(encoding))

        offset = recoveries.pop(0)
        end = encoding.char_to_token(batch_or_char_index=char_span[1] + offset)

    span = TokenSpan(start, end+1)
    return span


def get_entity_summaries(tok_spans: List[List[Tuple[int]]], support_embeddings: List[Tensor], summariser, query_vec=None, type=ENTITY):
    flat_spans = []
    flat_vecs = []
    for s, spans in enumerate(tok_spans):  # for each support document
        flat_spans.extend(spans)
        flat_vecs.extend([support_embeddings[s]] * len(spans))
    # return [summariser(vec, ENTITY, flat_spans[i]) for i, vec in enumerate(flat_vecs)]
    return summariser(flat_vecs, type, flat_spans, query_vec=query_vec)


def get_transformer_entity_token_spans(support_encodings, supports, get_sentence_spans=False, tokeniser=None) -> List[List[Tuple[int]]]:
    """
        token_spans is indexed list[support_no][ent_no]
        summaries is a flat list
    """
    token_spans: List[List[Tuple[int]]] = []

    for s, support in enumerate(supports):
        """get entity node embeddings"""
        if not get_sentence_spans:
            ent_c_spans = get_entity_char_spans(support)
        else:
            ent_c_spans = get_sentence_char_spans(support)

        support_encoding: BatchEncoding = support_encodings[s]

        ent_token_spans: List[Tuple[int]] = []
        for e, c_span in enumerate(ent_c_spans):
            """clips out the entities token embeddings, and summarises them"""
            try:
                ent_token_span = charspan_to_tokenspan(support_encoding, c_span)
            except Exception as ex:
                print("cannot get ent", e, "token span. in supp", s)
                text = tokeniser.decode(support_encoding["input_ids"])
                print("text:", text)
                print("text len:", len(text))
                print("char slice:", text[c_span[0]: c_span[1]])
                print(ex)
                continue
            ent_token_spans.append(ent_token_span)

        token_spans.append(ent_token_spans)

    return token_spans


def connect_sentence_and_entity_nodes(graph, tokeniser: LongformerTokenizerFast=None,
                                      support_encodings: List[BatchEncoding]=None, glove_embedder: GloveEmbedder=None):
    """
        Adds sentence nodes if they contain any entity nodes.
        Connects the sentence nodes to their documents and all entities whose text is contained in the sentences
        Sentences are thus allowed to connect to entities from different documents
    """
    all_sentence_nodes = get_entity_nodes(graph.example.supports, graph.example.sent_token_spans, tokeniser=tokeniser,
                                          support_encodings=support_encodings, glove_embedder=glove_embedder, get_sentence_spans=True)

    ent_nodes = [graph.ordered_nodes[i] for i in graph.entity_nodes]
    sentence_inclusion_bools = []
    for d, sentence_nodes in enumerate(all_sentence_nodes):
        sup_node = graph.get_doc_nodes()[d]

        for sent_node in sentence_nodes:

            sent_id = None
            for ent_node in ent_nodes:
                if clean(ent_node.text) in clean(sent_node.text):
                    # print("found ent",  ent_node.text, " in sent:", sent_node.text)
                    if sent_id is None:  # add sentence node to graph
                        sent_id = graph.add_node(sent_node)
                        doc_edge = HDEEdge(sup_node.id_in_graph, sent_id, graph=graph)
                        graph.add_edge(doc_edge)
                        # print("adding sentence node", sent_node)
                        # print("since it contains ent:", ent_node)
                    ent_edge = HDEEdge(sent_id, ent_node.id_in_graph, graph=graph)
                    graph.add_edge(ent_edge)

            sentence_inclusion_bools.append(sent_id is not None)
    graph.sentence_inclusion_bools = sentence_inclusion_bools

    print("num ents:", len(ent_nodes), "num sents:", len(graph.sentence_nodes))


def create_graph(example: Wikipoint, glove_embedder=None, tokeniser=None, support_encodings=None):
    graph = HDEGraph(example)
    add_doc_nodes(graph, example.supports)
    if tokeniser is not None and support_encodings is None:
        support_encodings = [tokeniser(support) for support in example.supports]
    add_entity_nodes(graph, example.supports, example.ent_token_spans, glove_embedder=glove_embedder,
                     tokeniser=tokeniser, support_encodings=support_encodings)

    add_candidate_nodes(graph, example.candidates, example.supports)
    connect_candidates_and_entities(graph)
    connect_entity_mentions(graph)

    if hasattr(conf, "use_sentence_nodes") and conf.use_sentence_nodes:  # todo remove legacy
        connect_sentence_and_entity_nodes(graph, glove_embedder=glove_embedder,
                         tokeniser=tokeniser, support_encodings=support_encodings)

    if conf.visualise_graphs:
        if conf.exit_after_first_viz:
            render_graph2(graph, graph_folder="temp")
            exit()
        else:
            cands = sorted([c.lower() for c in example.candidates])
            hash_code = zlib.adler32(str.encode("_".join(cands)))
            name = "graph_" + str(hash_code)
            render_graph2(graph, view=False, graph_name=name)

            text_name = name + ".txt"
            text_path = get_file_path(conf.model_name, text_name)
            file = open(text_path, "w", encoding='utf-8')

            type_counts = {}
            for edge in graph.ordered_edges:
                if edge.type() not in type_counts:
                    type_counts[edge.type()] = 0
                type_counts[edge.type()] += 1

            num_nodes = len(graph.ordered_nodes)
            num_edges = len(graph.unique_edges)
            num_cross_doc_ments = len(graph.get_cross_document_comention_edges())

            file.write(repr(example) + '\n\n'+
                       "edges: " + repr(num_edges) + "\n"+
                       "nodes: " + repr(num_nodes) + "\n\n" +
                       "edge density: " + repr(num_edges / (num_nodes * (num_nodes-1))) + "\n" +
                       "cross doc comentions: " + repr(num_cross_doc_ments) + "\n" +
                       "cross doc ratio: " + repr(num_cross_doc_ments / len(graph.doc_nodes)) + "\n" +

                       "edge types:\n\t" + "\n\t".join([t + ": " + repr(c) for t, c in type_counts.items()])
                       )

            file.close()

    return graph


def connect_all_to_all(source_node_ids: List[int], target_node_ids: List[int], graph, type):
    for source_node_id in source_node_ids:
        connect_one_to_all(source_node_id, target_node_ids, graph, type)


def connect_one_to_all(source_node_id: int, target_node_ids: List[int], graph, type=None):
    for target_id in target_node_ids:
        if source_node_id == target_id:  # no self loops
            continue
        edge = HDEEdge(source_node_id, target_id, type=type, graph=graph)
        graph.safe_add_edge(edge)


def fully_connect(node_ids, graph, type):
    connect_all_to_all(node_ids, node_ids, graph, type)


