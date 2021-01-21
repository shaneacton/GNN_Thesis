from typing import List, Tuple

import torch
from torch import nn
from torch.nn import ModuleList, ReLU, CrossEntropyLoss
from torch_geometric.nn import GATConv, MessagePassing
from transformers import BatchEncoding, LongformerTokenizerFast

from Code.Config import gcc
from Code.Data.Text.longformer_embedder import LongformerEmbedder
from Code.HDE.graph import HDEGraph
from Code.HDE.graph_utils import add_doc_nodes, add_entity_nodes, get_entities, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions
from Code.HDE.summariser import Summariser
from Code.Training import device
from Code.Training.Utils.initialiser import get_tokenizer


class HDE(nn.Module):

    def __init__(self, num_layers=2, hidden_size=402):
        super().__init__()
        self.tokeniser: LongformerTokenizerFast = get_tokenizer()
        self.token_embedder = LongformerEmbedder(out_features=hidden_size)

        self.summariser = Summariser(hidden_size)
        self.relu = ReLU()
        gnn_layers: List[MessagePassing] = []
        for i in range(num_layers):
            gnn_layers.append(GATConv(hidden_size, hidden_size))

        self.gnn_layers = ModuleList(gnn_layers)

        self.cand_prob_map = nn.Linear(hidden_size, 1)
        self.loss_fn = CrossEntropyLoss()


    def forward(self, supports: List[str], query: str, candidates: List[str], answer=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """

        support_encodings, candidate_encodings = self.get_encodings(supports, query, candidates)

        support_embeddings = [self.token_embedder(sup_enc) for sup_enc in support_encodings]

        candidate_summaries = [self.summariser(self.token_embedder(cand_enc, all_global=True)) for cand_enc in candidate_encodings]
        support_summaries = [self.summariser(sup_emb) for sup_emb in support_embeddings]

        ent_token_spans, ent_summaries, = get_entities(self.summariser, support_embeddings, support_encodings, supports)

        graph = self.create_graph(candidates, ent_token_spans, support_encodings, supports)

        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        # print("x:", x.size())

        edge_index = graph.edge_index
        # print("edge index:", edge_index.size(), edge_index.type())

        for layer in self.gnn_layers:
            x = self.relu(layer(x, edge_index=edge_index))

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

    def create_graph(self, candidates, ent_token_spans, support_encodings, supports):
        graph = HDEGraph()
        add_doc_nodes(graph, supports)
        add_entity_nodes(graph, supports, support_encodings, ent_token_spans, self.tokeniser)
        add_candidate_nodes(graph, candidates, supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)
        return graph

    def get_encodings(self, supports: List[str], query: str, candidates: List[str]) \
            -> Tuple[List[BatchEncoding], List[BatchEncoding]]:
        supports = [s[:gcc.max_context_chars] for s in supports]
        support_encodings = [self.tokeniser(support, query) for support in supports]
        candidate_encodings = [self.tokeniser(candidate) for candidate in candidates]
        return support_encodings, candidate_encodings
