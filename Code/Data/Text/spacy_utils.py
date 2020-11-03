from typing import List, Tuple, Dict

import spacy
from neuralcoref.neuralcoref import Cluster
from spacy.tokens.span import Span
from spacy.tokens.token import Token

nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe

added_neuralcoref = False


def init_doc(doc, text):
    if not doc:
        doc = nlp(text)
    return doc


def get_char_span_from_spacy_span(span: Span, doc) -> Tuple[int]:
    start_char = doc[span.start].idx
    end_char = doc[span.end -1].idx + len(doc[span.end -1])
    return start_char, end_char


def get_sentences(text, doc=None):
    doc = init_doc(doc, text)
    sents = doc.sents


def get_coref_char_spans(doc) -> Dict[Tuple[int], List[Tuple[int]]]:
    corefs: List[Cluster] = doc._.coref_clusters

    coref_chars: Dict[Tuple[int], List[Tuple[int]]] = {}
    for cor in corefs:
        """get the char spans of the ents with corefs, as well as their corefs"""
        main_char_span = get_char_span_from_spacy_span(cor.main, doc)
        coref_spans = [get_char_span_from_spacy_span(men, doc) for men in cor.mentions]
        coref_spans = coref_spans[1:]  # cut off main ent from corefs
        coref_chars[main_char_span] = coref_spans

    return coref_chars


def get_entity_and_coref_chars(text, doc=None) -> List[Tuple[Tuple[int], List[Tuple[int]]]]:
    global added_neuralcoref
    global nlp

    if not added_neuralcoref:
        import neuralcoref
        neuralcoref.add_to_pipe(nlp)
        doc = None  # must reinit doc with NC
    doc = init_doc(doc, text)

    coref_chars = get_coref_char_spans(doc)
    ent_and_coref_chars: List[Tuple[Tuple[int], List[Tuple[int]]]] = []
    for ent in doc.ents:
        """get the char spans of all ents, combine with corefs if available"""
        ent_span = get_char_span_from_spacy_span(ent, doc)
        # print("ent:", text[ent_span[0]: ent_span[1]])
        cor_span = coref_chars[ent_span] if ent_span in coref_chars else []
        ent_and_coref_chars.append((ent_span, cor_span))

    print(ent_and_coref_chars)
    for e in ent_and_coref_chars:
        # print(e)
        print(text[e[0][0]:e[0][1]])
        corefs = [text[c[0]: c[1]] for c in e[1]]
        print(corefs)

def nouns(text, doc=None):
    doc = init_doc(doc, text)
    nouns = doc.noun_chunks

if __name__ == "__main__":
    sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very " \
               "close to the Manhattan Bridge which is visible from the window."
    print(sequence)
    print(get_entity_and_coref_chars(sequence))