from typing import List, Tuple

from torch import nn, Tensor
from torch.nn import ModuleList, ReLU
from torch_geometric.nn import GATConv
from transformers import TokenSpan, BatchEncoding

from Code.Data.Text.longformer_embedder import LongformerEmbedder
from Code.Data.Text.spacy_utils import get_entity_char_spans
from Code.HDE.summariser import Summariser
from Code.Training.Utils.initialiser import get_tokenizer
import numpy as np


class HDE(nn.Module):

    def __init__(self, num_layers=2, hidden_size=402):
        super().__init__()
        self.tokeniser = get_tokenizer()
        self.token_embedder = LongformerEmbedder(out_features=hidden_size)

        self.summariser = Summariser(hidden_size)
        gnn_layers = []
        for i in range(num_layers):
            gnn_layers.append(GATConv(hidden_size, hidden_size))
            gnn_layers.append(ReLU())

        self.gnn_layers = ModuleList(gnn_layers)

    def forward(self, supports: List[str], query: str, candidates: List[str]):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates
        """

        support_encodings, candidate_encodings = self.get_encodings(supports, query, candidates)

        support_embeddings = [self.token_embedder(sup_enc) for sup_enc in support_encodings]

        candidate_summaries = [self.summariser(self.token_embedder(cand_enc)) for cand_enc in candidate_encodings]
        support_summaries = [self.summariser(sup_emb) for sup_emb in support_embeddings]

        ent_summaries, ent_token_spans = self.get_entities(support_embeddings, support_encodings, supports)
        flat_ent_summaries = np.array(ent_summaries).flatten()

        candidate_containment = []  # indexed cc[support][containment_count]

        for s, support in enumerate(supports):
            candidates_contained = []
            for c, cand in enumerate(candidates):
                """
                    finds which candidates are contained in which supports
                    corresponds to edge type 1 in HDE paper
                """
                if cand in support:
                    candidates_contained.append(c)
            candidate_containment.append(candidates_contained)



    def get_entities(self, support_embeddings, support_encodings, supports):
        """
        :return: both 2d lists are indexed list[support_no][ent_no]
        """
        token_spans: List[List[Tuple[int]]] = []
        summaries: List[List[Tensor]] = []

        for s, support in enumerate(supports):
            """get entity node embeddings"""
            ent_c_spans = get_entity_char_spans(support)
            support_encoding = support_encodings[s]

            ent_summaries: List[Tensor] = []
            ent_token_spans: List[Tuple[int]] = []
            for e, c_span in enumerate(ent_c_spans):
                """clips out the entities token embeddings, and summarises them"""
                ent_token_span = self.charspan_to_tokenspan(support_encoding, c_span)
                ent_token_spans.append(ent_token_span)
                ent_summaries.append(self.summariser(support_embeddings, ent_token_span))

            token_spans.append(ent_token_spans)
            summaries.append(ent_summaries)

        return token_spans, summaries

    def charspan_to_tokenspan(self, encoding, char_span: Tuple[int]) -> TokenSpan:
        start = self.encoding.char_to_token(char_index=char_span[0], batch_or_char_index=self.batch_id)

        recoveries = [-1, 0, -2, -3]  # which chars to try. To handle edge cases such as ending on dbl space ~ '  '
        end = None
        while end is None:
            if len(recoveries) == 0:
                raise Exception(
                    "could not get end token span from char span:" + repr(char_span) + " num tokens: " + repr(
                        len(self.encoding.tokens())) + " ~ " + repr(self.encoding))

            offset = recoveries.pop(0)
            end = self.encoding.char_to_token(char_index=char_span[1] + offset, batch_or_char_index=self.batch_id)

        span = TokenSpan(start - 1, end)  # -1 to discount the <s> token
        return span

    def get_encodings(self, supports: List[str], query: str, candidates: List[str]) \
            -> Tuple[List[BatchEncoding], List[BatchEncoding]]:

        support_encodings = [self.tokeniser(support, query) for support in supports]
        candidate_encodings = [self.tokeniser(candidate) for candidate in candidates]
        return support_encodings, candidate_encodings
