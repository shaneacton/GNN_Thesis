import stanfordnlp
import torch
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import WordTokenizer
from transformers import BertTokenizer, BertModel

from Code.GNN_Playground.Models import embedder
from Code.GNN_Playground.Models.Vanilla.bidaf import BiDAF
from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader
from Datasets.Readers.squad_reader import SQuADDatasetReader

text = "the apple is bad. super bad. I even thought it was ok." * 30

tokeniser = WordTokenizer()

squad_reader = SQuADDatasetReader(tokeniser)
qangaroo_reader = QUangarooDatasetReader(tokeniser)

bidaf = BiDAF(100)

for training_example in squad_reader.get_training_examples(SQuADDatasetReader.dev_set_location()):
    print(training_example)
    output = bidaf(training_example.context, training_example.questions[0])
    print("squad out:", [o.size() for o in output])
    break

# for training_example in qangaroo_reader.get_training_examples(QUangarooDatasetReader.dev_set_location("wikihop")):
#     # print(training_example)
#     training_example.context.get_context_embedding()
#     output = bidaf(training_example.context, training_example.questions[0])
#     print("wikihop out:", [o.size() for o in output])
#     break

#
# squad_vocab = Vocabulary.from_instances(squad_reader.read(squad_reader.dev_set_location()))
# wikihop_vocab = Vocabulary.from_instances(qangaroo_reader.read(qangaroo_reader.dev_set_location("wikihop")))
#
# token_to_id = wikihop_vocab.get_token_to_index_vocabulary()
# tokens = tokeniser.tokenize(text)
# print(tokens)
# ids = [token_to_id[token.text] if (token.text in token_to_id.keys()) else -1 for token in tokens]
# print(ids)
#
# translation = {token.text: token_to_id[token.text] if (token.text in token_to_id.keys()) else -1 for token in tokens}
# print(translation)

embeddings = embedder(text)
print("bert embeddings:", embeddings, "shape:", embeddings.size())

# stanfordnlp.download('en')
# nlp = stanfordnlp.Pipeline(processors="tokenize,mwt,lemma,pos")
# doc = nlp(
#     """The prospects for Britain’s orderly withdrawal from the European Union on March 29 have receded further,
#     even as MPs rallied to stop a no-deal scenario. An amendment to the draft bill on the termination of London’s
#     membership of the bloc obliges Prime Minister Theresa May to renegotiate her withdrawal agreement with Brussels.
#     A Tory backbencher’s proposal calls on the government to come up with alternatives to the Irish backstop,
#     a central tenet of the deal Britain agreed with the rest of the EU.""")
#
# # doc.sentences[0].print_dependencies()
#
# for sentence in doc.sentences:
#     for word in sentence.words:
#         print("word:",word,"pos:",word.pos)
