import time

import torch
from transformers import BertModel, BasicTokenizer

from Code.Data.Text.Tokenisation.bert_tokeniser import CustomBertTokenizer
from Code.Training import device


embedder_type = "bert"

if embedder_type == "bert":
    print("initialising bert")
    start = time.time()
    basic_bert_tokeniser = BasicTokenizer()
    bert_tokeniser = CustomBertTokenizer.from_pretrained("bert-base-uncased")

    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    basic_tokeniser = lambda string: basic_bert_tokeniser.tokenize(string)  # no subtoken splitting
    subtoken_mapper = lambda string: bert_tokeniser.tokenize(string)[1]

    tokeniser = lambda string: bert_tokeniser.tokenize(string)[0]
    token_indexer = lambda tokens: torch.Tensor(bert_tokeniser.convert_tokens_to_ids(tokens)).reshape(1, -1) \
        .type(torch.LongTensor).to(device)
    token_embedder = lambda tokens: bert_model(token_indexer(tokens))[0]

    print("bert initialised in", (time.time()-start), "secs")


indexer = lambda info: string_indexer(info) if isinstance(info,str) else token_indexer(info)
string_embedder = lambda string: token_embedder(tokeniser(string))
string_indexer = lambda string: token_indexer(tokeniser(string))


embedder = lambda info: string_embedder(info) if isinstance(info,str) else token_embedder(info)
embedded_size = list(embedder("test").size())[2]

def tail_concatinator(seq_embedding, cat_dim=2):
    seq_head = seq_embedding[:,0:1,:]
    num_el = seq_embedding.size(1)
    if num_el == 1:
        return torch.cat([seq_head,seq_head], dim=cat_dim)

    return torch.cat([seq_head,seq_embedding[:,num_el-1:num_el,:]], dim=cat_dim)


def sum_over_sequence(seq_embedding):
    batch_size = seq_embedding.size(0)
    return torch.sum(seq_embedding, dim=1).view(batch_size,1,-1)

