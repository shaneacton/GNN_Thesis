from torch import nn


class ContextualEmbedder(nn.Module):
    """
    wrapper around any pretrained contextual embedder such as bert
    """

