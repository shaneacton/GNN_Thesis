import torch
from transformers import BertTokenizer, BertModel

embedder_type = "bert"

if embedder_type == "bert":

    bert_tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    tokeniser = lambda string: bert_tokeniser.tokenize(string)

    token_indexer = lambda tokens: torch.Tensor(bert_tokeniser.convert_tokens_to_ids(tokens)).reshape(1, -1).type(torch.LongTensor)

    token_embedder = lambda tokens: bert_model.embeddings(token_indexer(tokens))


string_indexer = lambda string: token_indexer(tokeniser(string))
string_embedder = lambda string: token_embedder(tokeniser(string))

embedder = lambda info: string_embedder(info) if isinstance(info,str) else token_embedder(info)
indexer = lambda info: string_indexer(info) if isinstance(info,str) else token_indexer(info)

embedded_size = list(embedder("test").size())[2]
# print("embedding size:",embedded_size)