from typing import Type

import torch
from transformers import PreTrainedModel

import Code.Data.Text.text_utils
from Code.Config import gec
from Code.Data.Text.Answers.candidate_answer import CandidateAnswer
from Code.Data.Text.data_sample import DataSample
from Code.Data.Text.pretrained_token_sequence_embedder import PretrainedTokenSequenceEmbedder
from Code.Models.GNNs.OutputModules.node_selection import NodeSelection
from Code.Models.context_nn import ContextNN
from Code.Training import device


class ContextTransformer(ContextNN):

    def __init__(self, model_type: Type, configuration, num_layers, heads=8):
        super().__init__()
        self.configuration = configuration

        token_seq_embedder = PretrainedTokenSequenceEmbedder(gec)
        self.indexer = lambda string: token_seq_embedder.index(token_seq_embedder.tokenise(string))

        vocab_size = len(token_seq_embedder.bert_tokeniser.vocab)
        configuration.vocab_size = vocab_size
        configuration.max_position_embeddings = 4000
        configuration.type_vocab_size = 3
        configuration.num_attention_heads = heads
        configuration.num_hidden_layers = num_layers

        self.transformer: PreTrainedModel = model_type(configuration)

    def get_output_model_type(self, data_sample: DataSample):
        answer_type = data_sample.get_answer_type()
        if answer_type == CandidateAnswer:
            return NodeSelection
        return ContextNN.get_output_model_type(self, data_sample)

    def forward(self, batch):
        ids, type_ids, global_attention_mask, output_ids = self.get_ids(batch)
        encoded, _ = self.transformer(input_ids=ids, token_type_ids=type_ids, global_attention_mask=global_attention_mask)
        output_probs = self.output_model(encoded, output_ids=output_ids)
        return output_probs

    def get_ids(self, batch):
        """
            gets the token and type ids, as well as global attention mask
            if batch has candidate answers, returns: cands, sep, query, sep, context
            else returns context, sep, query
        """
        q = Code.Data.Text.text_utils.question
        q_ids = self.indexer(q.raw_text)
        ctx_ids = self.indexer(batch.data_sample.context.get_full_context())
        sep_id = torch.tensor(self.configuration.sep_token_id).view(1, 1)
        # print("ctx:", ctx_ids.size(), "q:", q_ids.size(), "sep:", sep_id)

        if batch.has_candidates:
            cands = [cand.raw_text for cand in q.answers.answer_candidates]
            cands = [self.indexer(cand) for cand in cands]
            output_ids = []
            s = 0
            for cand in cands:
                output_ids.append(s)
                s += cand.size(1)
            cand_ids = torch.cat(cands, dim=1)
            # print("cands:", cand_ids.size(), "cand starts:", starts)

            ids = torch.cat([cand_ids, sep_id, q_ids, sep_id, ctx_ids], dim=1)
            type_ids = torch.cat([torch.tensor([0] * cand_ids.size(1)), torch.tensor([1] * (q_ids.size(1) + 2)),
                                  torch.tensor([2] * ctx_ids.size(1))])
            global_ids = list(range(cand_ids.size(1) + 1 + q_ids.size(1)))  # all cands and q's
        else:  # span selection
            ids = torch.cat([ctx_ids, sep_id, q_ids], dim=1)
            type_ids = torch.cat([torch.tensor([0] * (ctx_ids.size(1) + 1)), torch.tensor([1] * q_ids.size(1))])
            global_ids = list(range(ctx_ids.size(1), q_ids.size(1)))  # all q's

            output_ids = list(range(ctx_ids.size(1)))  # any of the context tokens are valid outputs

        output_ids = torch.tensor(output_ids).view(1, -1)

        global_attention_mask = torch.zeros(ids.shape, dtype=torch.long, device=device)
        global_attention_mask[:, global_ids] = 1

        return ids, type_ids, global_attention_mask, output_ids

