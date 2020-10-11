from Code.Data.Graph.Embedders.position_embedder import PositionEmbedder


class RelativePositionEmbedder(PositionEmbedder):

    def __init__(self, num_features, gcc, gec):
        super().__init__(num_features, gcc, gec)

    def get_expected_position_ids(self, level):
        window_size = self.gec.relative_embeddings_window_per_level[level]
        return range(-window_size, window_size)
