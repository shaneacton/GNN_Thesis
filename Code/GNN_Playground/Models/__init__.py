import torch
from transformers import BertTokenizer, BertModel

embedder_type = "bert"

if embedder_type == "bert":

    bert_tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    tokeniser = lambda string: bert_tokeniser.tokenize(string)
    indexer = lambda string: torch.Tensor(bert_tokeniser.convert_tokens_to_ids(tokeniser(string))).reshape(1, -1).type(torch.LongTensor)
    embedder = lambda string: bert_model.embeddings(indexer(string))


embedded_size = list(embedder("test").size())[2]
# print("embedding size:",embedded_size)