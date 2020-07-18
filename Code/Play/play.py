import nltk
import stanfordnlp
# import neuralcoref
from nltk import pos_tag, ne_chunk

from Code.Data.Text.text import Text

raw_text = "the apple is bad. super bad. I even thought it was ok. My  name is Edmandolo" * 1
text = Text(raw_text)

print("clean:",text.clean)
print("tokens:", text.token_sequence.raw_tokens)
print("subtoken map:", text.token_sequence.raw_subtokens)

nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

print("nltk pos:", pos_tag(text.token_sequence.raw_tokens))
print("nlkt NER:", ne_chunk(pos_tag(text.token_sequence.raw_tokens)))

# stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline(processors="tokenize,mwt,lemma,pos")
doc = nlp(text.clean)

doc.sentences[0].print_dependencies()

for sentence in doc.sentences:
    for word in sentence.words:
        print("word:",word,"pos:",word.pos)



import torch
from torch_geometric.data import Data


x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)


data = Data(x=x, y=y, edge_index=edge_index)