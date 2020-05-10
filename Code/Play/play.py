import nltk
import stanfordnlp
import neuralcoref
from nltk import pos_tag, ne_chunk


from Code.Data.text import Text

raw_text = "the apple is bad. super bad. I even thought it was ok. My  name is Edmandolo" * 1
text = Text(raw_text)

print("clean:",text.clean)
print("tokens:", text.token_sequence.tokens)
print("subtoken map:", text.token_sequence.sub_tokens)

nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

print("nltk pos:",pos_tag(text.token_sequence.tokens))
print("nlkt NER:",ne_chunk(pos_tag(text.token_sequence.tokens)))

# stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline(processors="tokenize,mwt,lemma,pos")
doc = nlp(text.clean)

doc.sentences[0].print_dependencies()

for sentence in doc.sentences:
    for word in sentence.words:
        print("word:",word,"pos:",word.pos)
