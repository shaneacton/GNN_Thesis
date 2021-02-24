import time

from Code.Embedding.bert_embedder import BertEmbedder
from Code.HDE.hde_model import HDEModel
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.graph_utils import add_doc_nodes, add_entity_nodes, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions
from Config.config import conf
from Viz.graph_visualiser import render_graph


class HDEBert(HDEModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedder = BertEmbedder()
        self.embedder_name = "bert"

    def create_graph(self, example):
        start_t = time.time()
        graph = HDEGraph()
        add_doc_nodes(graph, example.supports)
        add_entity_nodes(graph, example.supports, example.ent_token_spans, tokeniser=self.embedder.tokenizer)
        add_candidate_nodes(graph, example.candidates, example.supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)

        if conf.print_times:
            print("made full graph in", (time.time() - start_t))

        if conf.visualise_graphs:
            render_graph(graph)
            if conf.exit_after_first_viz:
                exit()

        return graph