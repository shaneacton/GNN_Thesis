import copy
from typing import List, Tuple, Dict

import spacy
from neuralcoref.neuralcoref import Cluster
from spacy.tokens.span import Span

from Code.Data.Text.text_utils import context
from Code.Test.examples import test_example

try:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
except:
    print("failed to load en_core_web_sm, trying in clisyer location")
    import spacy
    spacy.util.set_data_path('/home/sacton/.conda/envs/gnn_env/lib/python3.8/site-packages')
    nlp = spacy.load('en_core_web_sm')
# Add neural coref to SpaCy's pipe

added_neuralcoref = False


def _init_doc(doc, text):
    if not doc:
        doc = nlp(text)
    return doc


def get_char_span_from_spacy_span(span: Span, doc) -> Tuple[int]:
    start_char = doc[span.start].idx
    end_char = doc[span.end -1].idx + len(doc[span.end -1])
    return start_char, end_char


def get_sentence_char_spans(text, doc=None) -> List[Tuple[int]]:
    doc = _init_doc(doc, text)
    sents = doc.sents
    sent_spans = [get_char_span_from_spacy_span(s, doc) for s in sents]

    return sent_spans, doc


def _get_coref_char_spans(doc) -> Dict[Tuple[int], List[Tuple[int]]]:
    corefs: List[Cluster] = doc._.coref_clusters
    print(corefs)

    coref_chars: Dict[Tuple[int], List[Tuple[int]]] = {}
    for cor in corefs:
        """get the char spans of the ents with corefs, as well as their corefs"""
        main_char_span = get_char_span_from_spacy_span(cor.main, doc)
        coref_spans = [get_char_span_from_spacy_span(men, doc) for men in cor.mentions]
        coref_spans = coref_spans[1:]  # cut off main ent from corefs
        coref_chars[main_char_span] = coref_spans

    return coref_chars


def get_entity_and_coref_chars(text, doc=None) -> List[Tuple[Tuple[int], List[Tuple[int]]]]:
    """returns a list of entity:[corefs] char spans"""
    global added_neuralcoref
    global nlp

    if not added_neuralcoref:
        import neuralcoref
        neuralcoref.add_to_pipe(nlp)
        doc = None  # must reinit doc with NC
    doc = _init_doc(doc, text)

    coref_chars = _get_coref_char_spans(doc)
    ent_and_coref_chars: List[Tuple[Tuple[int], List[Tuple[int]]]] = []
    for ent in doc.ents:
        """get the char spans of all ents, combine with corefs if available"""
        ent_span = get_char_span_from_spacy_span(ent, doc)
        # print("ent:", text[ent_span[0]: ent_span[1]])
        cor_span = coref_chars[ent_span] if ent_span in coref_chars else []
        ent_and_coref_chars.append((ent_span, cor_span))

    return ent_and_coref_chars, doc


def get_flat_entity_and_corefs_chars(text, doc=None):
    ent_and_coref_chars, doc = get_entity_and_coref_chars(text, doc=doc)
    flat = [m[0] for m in ent_and_coref_chars]
    for m in ent_and_coref_chars:
        flat.extend(m[1])
    flat.sort(key=lambda span: (span[0] + span[1]) * 0.5)
    return flat, doc


def get_noun_char_spans(text, doc=None):
    doc = _init_doc(doc, text)
    nouns = [n for n in doc.noun_chunks]
    return [get_char_span_from_spacy_span(n, doc) for n in nouns], doc


if __name__ == "__main__":
    # text = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very " \
    #            "close to the Manhattan Bridge which is visible from the window."
    text = context(test_example)
    # print(sequence)
    doc = None
    # char_spans, doc = get_flat_entity_and_corefs_chars(text, doc=doc)
    # char_spans, doc = get_sentence_char_spans(text, doc=doc)
    char_spans, doc = get_noun_char_spans(text, doc=doc)

    for s in char_spans:
        print(text[s[0]: s[1]])