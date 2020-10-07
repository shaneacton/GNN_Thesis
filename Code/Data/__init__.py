import threading

#
#
# embedder_type = "bert"
#
# if embedder_type == "bert":
#     print("initialising bert")
#     start = time.time()
#     basic_bert_tokeniser = BasicTokenizer()
#     bert_tokeniser = BertTokenizer.from_pretrained("bert-base-uncased")
#
#     bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
#
#     basic_tokeniser = lambda string: basic_bert_tokeniser.tokenize(string)  # no subtoken splitting
#
#     tokeniser = lambda string: bert_tokeniser.tokenize(string)
#     token_indexer = lambda tokens: torch.Tensor(bert_tokeniser.convert_tokens_to_ids(tokens)).reshape(1, -1) \
#         .type(torch.LongTensor).to(device)
#     token_embedder = lambda tokens: bert_model(token_indexer(tokens))[0]
#
#     print("bert initialised in", (time.time()-start), "secs")
#
#
# indexer = lambda info: string_indexer(info) if isinstance(info,str) else token_indexer(info)
# string_embedder = lambda string: token_embedder(tokeniser(string))
# string_indexer = lambda string: token_indexer(tokeniser(string))
#
#
# embedder = lambda info: string_embedder(info) if isinstance(info,str) else token_embedder(info)
# embedded_size = list(embedder("test").size())[2]


