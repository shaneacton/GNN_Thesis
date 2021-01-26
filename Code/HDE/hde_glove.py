import time
from typing import List

import torch
from torch import nn
from torch.nn import ModuleList, ReLU, CrossEntropyLoss
from torch_geometric.nn import GATConv, MessagePassing

from Code.Config import sysconf
from Code.HDE.Glove.glove_embedder import GloveEmbedder
from Code.HDE.Glove.glove_utils import get_glove_entities
from Code.HDE.coattention import Coattention
from Code.HDE.graph import HDEGraph
from Code.HDE.graph_utils import add_doc_nodes, add_entity_nodes, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions
from Code.HDE.summariser import Summariser
from Code.Training import device


class HDEGloveEmbed(nn.Module):

    def __init__(self, num_layers=2, hidden_size=402):
        super().__init__()
        self.embedder = GloveEmbedder()

        self.coattention = Coattention(self.embedder.dims)
        self.summariser = Summariser(self.embedder.dims)
        self.relu = ReLU()
        gnn_layers: List[MessagePassing] = []
        for i in range(num_layers):
            if i == 0:
                gnn_layers.append(GATConv(self.embedder.dims, hidden_size))
            else:
                gnn_layers.append(GATConv(hidden_size, hidden_size))

        self.gnn_layers = ModuleList(gnn_layers)

        self.cand_prob_map = nn.Linear(hidden_size, 1)
        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1

    def forward(self, supports: List[str], query: str, candidates: List[str], answer=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        t = time.time()

        support_embeddings = self.get_query_aware_context_embeddings(supports, query)

        if sysconf.print_times:
            print("got embeddings in", (time.time() - t))
        t = time.time()

        candidate_summaries = [self.summariser(self.embedder(cand)) for cand in candidates]
        support_summaries = [self.summariser(sup_emb) for sup_emb in support_embeddings]

        if sysconf.print_times:
            print("got summaries in", (time.time() - t))
        t = time.time()

        ent_token_spans, ent_summaries, = get_glove_entities(self.summariser, support_embeddings, supports, self.embedder)

        if sysconf.print_times:
            print("got ents in", (time.time() - t))
        t = time.time()

        graph = self.create_graph(candidates, ent_token_spans, supports)

        if sysconf.print_times:
            print("made graph in", (time.time() - t))
        t = time.time()

        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        edge_index = graph.edge_index
        # print("edge index:", edge_index.size(), edge_index.type())

        for layer in self.gnn_layers:
            x = self.relu(layer(x, edge_index=edge_index))

        # print("passed gnn in", (time.time() - t))
        t = time.time()

        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates

        cand_idxs = graph.candidate_nodes
        cand_idxs = (cand_idxs[0], cand_idxs[-1])
        cand_embs = x[cand_idxs[0]: cand_idxs[1] + 1, :]

        probs = self.cand_prob_map(cand_embs).squeeze()
        pred_id = torch.argmax(probs)
        pred_ans = candidates[pred_id]

        if answer is not None:
            ans_id = candidates.index(answer)
            probs = probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)

            return loss, pred_ans

        return pred_ans

    def get_query_aware_context_embeddings(self, supports: List[str], query: str):
        support_embeddings = [self.embedder(sup) for sup in supports]
        query_emb = self.embedder(query)
        support_embeddings = [self.coattention(se, query_emb) for se in support_embeddings]
        return support_embeddings

    def create_graph(self, candidates, ent_token_spans, supports):
        graph = HDEGraph()
        add_doc_nodes(graph, supports)
        add_entity_nodes(graph, supports, ent_token_spans, glove_embedder=self.embedder)
        add_candidate_nodes(graph, candidates, supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)
        return graph
