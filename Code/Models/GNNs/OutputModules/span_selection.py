from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Models.loss_funcs import get_span_loss
from Code.constants import CONTEXT, TOKEN


class SpanSelection(OutputModel):

    def __init__(self, in_features):
        super().__init__(in_features)
        from Code.Models.GNNs.OutputModules.token_selection import TokenSelection

        self.start_selector = TokenSelection(in_features)
        self.end_selector = TokenSelection(in_features)

    def get_node_ids_from_graph(self, graph: QAGraph, **kwargs):
        return self.start_selector.get_node_ids_from_graph(graph, **kwargs)

    def get_output_from_graph_encoding(self, data: GraphEncoding, **kwargs):
        if TOKEN not in data.sample_graph.gcc.structure_levels[CONTEXT]:
            raise Exception("cannot do span prediction without including context tokens")
        # print("span selection, kwargs:", kwargs)
        logits = self.start_selector(data, **kwargs), self.end_selector(data, **kwargs)
        if "start_positions" in kwargs and "end_positions" in kwargs:
            positions = kwargs["start_positions"], kwargs["end_positions"]
            loss = get_span_loss(positions[0], positions[1], logits[0], logits[1])

            # print("num prediction options:", logits[0].size(), "ans pos:", positions)
            return (loss,) + logits
        return logits