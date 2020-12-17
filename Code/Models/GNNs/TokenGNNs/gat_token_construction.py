import torch
from torch import nn, Tensor
from torch_geometric.nn import GATConv
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel

from Code.Config import GraphConstructionConfig, GraphEmbeddingConfig
from Code.Data.Graph.Contructors.qa_graph_constructor import QAGraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Text.longformer_embedder import LongformerEmbedder
from Code.Data.Text.text_utils import is_batched, get_single_example
from Code.Models.GNNs.ContextGNNs.context_gnn import prep_input
from Code.Training.Utils.initialiser import get_tokenizer
from Code.Training.Utils.text_encoder import TextEncoder
from Code.Training import device
from Code.constants import TOKEN, QUERY, CONTEXT

MAX_NODES = 2900  # 2900


class GatTokenConstruction(nn.Module):

    """
        sanity check which uses the configurable graph embedding system with tokens only
        uses the same output as the GatWraps which work
    """

    def __init__(self, _, output):
        super().__init__()
        self.middle_size = output.config.hidden_size
        long_embedder = LongformerEmbedder()
        emb_size = long_embedder.longformer.config.hidden_size
        # super().__init__(long_embedder.longformer.config)
        self.long_embedder = long_embedder
        self.middle1 = GATConv(emb_size, self.middle_size)
        self.act = nn.ReLU()
        self.middle2 = GATConv(self.middle_size, self.middle_size)

        self.output = output
        self.max_pretrained_pos_ids = self.long_embedder.longformer.config.max_position_embeddings
        print("max pos embs:", self.max_pretrained_pos_ids)

        self.token_construction_config = GraphConstructionConfig()
        self.token_construction_config.structure_levels = {CONTEXT: [TOKEN], QUERY: [TOKEN]}

        self.graph_embedder: GraphEmbedder = GraphEmbeddingConfig().get_graph_embedder(self.token_construction_config)
        self.graph_constructor = QAGraphConstructor(self.token_construction_config)

        self.text_encoder = TextEncoder(get_tokenizer())

    def forward(self, input, return_dict=True, **kwargs):
        input, kwargs = prep_input(input, kwargs)
        if is_batched(input):
            input = get_single_example(input)
        graph = self.graph_constructor(input)
        graph_embedding = self.graph_embedder(graph)
        node_embs = graph_embedding.x
        token_encoding = self.text_encoder.get_encoding(input)
        input_ids = torch.tensor(token_encoding["input_ids"]).to(device).view(1, -1)
        attention_mask = torch.tensor(token_encoding["attention_mask"]).to(device).view(1, -1)
        # long_embs = self.long_embedder.embed(input_ids, attention_mask).squeeze()[1:-1, :]

        # print("input:", input)
        # print("graph embedding:", graph_embedding)
        # print("nodes:", node_embs.size(), "\n", node_embs)
        # print("token embedding:", long_embs.size(), "\n", long_embs)
        ctx_len = self.long_embedder.get_first_sep_index(input_ids)
        # +1 for the fake sep token inserted at the beginning
        global_attention_mask = self.long_embedder.get_glob_att_mask_from(ctx_len + 1, node_embs.shape[0] + 1)

        edge = graph_embedding.edge_index
        node_embs = self.act(self.middle1(x=node_embs, edge_index=edge))
        node_embs = self.act(self.middle2(x=node_embs, edge_index=edge))
        node_embs = torch.cat([torch.zeros(1, self.middle_size).to(device), node_embs], dim=0).view(1, -1, self.middle_size)
        # print("after gat:", node_embs.size())
        # raise Exception("gtc weh")
        start_positions = input.pop("start_positions", None)
        end_positions = input.pop("end_positions", None)
        if "start_positions" in kwargs:
            start_positions = kwargs["start_positions"]
            end_positions = kwargs["end_positions"]
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)

        out = self.output(inputs_embeds=node_embs, return_dict=return_dict,
                          start_positions=start_positions, end_positions=end_positions,
                          global_attention_mask=global_attention_mask)
        return out

    def get_null_return(self, input_ids:Tensor, return_dict:bool, include_loss:bool):
        loss = None
        num_ids = input_ids.size(1)
        if include_loss:
            loss = torch.tensor(0., requires_grad=True)
        logits = torch.tensor([0.] * num_ids, requires_grad=True).to(float)

        if not return_dict:
            output = (logits, logits)
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=logits,
            end_logits=logits,
        )
