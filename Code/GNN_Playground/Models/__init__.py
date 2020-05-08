import torch
from transformers import BertTokenizer, BertModel, BasicTokenizer

from Code.GNN_Playground.Data.Tokenisation.bert_tokeniser import CustomBertTokenizer
from Code.GNN_Playground.Training import device

embedder_type = "bert"

if embedder_type == "bert":

    basic_bert_tokeniser = BasicTokenizer()
    bert_tokeniser = CustomBertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

    subtoken_mapper = lambda string: bert_tokeniser.tokenize(string)[1]
    tokeniser = lambda string: bert_tokeniser.tokenize(string)[0]

    token_indexer = lambda tokens: torch.Tensor(bert_tokeniser.convert_tokens_to_ids(tokens)).reshape(1, -1)\
        .type(torch.LongTensor).to(device)

    token_embedder = lambda tokens: bert_model.embeddings(token_indexer(tokens))


string_indexer = lambda string: token_indexer(tokeniser(string))
string_embedder = lambda string: token_embedder(tokeniser(string))

embedder = lambda info: string_embedder(info) if isinstance(info,str) else token_embedder(info)
indexer = lambda info: string_indexer(info) if isinstance(info,str) else token_indexer(info)

embedded_size = list(embedder("test").size())[2]
# print("embedding size:",embedded_size)