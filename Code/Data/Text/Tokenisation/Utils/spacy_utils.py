from typing import List

from spacy.tokens.doc import Doc

import Code.constants
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan

try:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
except:
    print("failed to load en_core_web_sm, trying in cluster location")
    import spacy
    spacy.util.set_data_path('/home/sacton/.conda/envs/gnn_env/lib/python3.8/site-packages')
    nlp = spacy.load('en_core_web_sm')

loaded_neuralcoref = False

last_processed_doc = None


def get_processed(tok_seq):
    global last_processed_doc

    if not last_processed_doc or tok_seq not in last_processed_doc:
        # not cached, must process
        last_processed_doc = {tok_seq: nlp(tok_seq.text_obj.raw_text)}

    return last_processed_doc[tok_seq]


def get_spacy_entities(tok_seq, spacy_processed_doc=None) -> List[EntitySpan]:
    """
    uses the unprocessed text to perform NER using Spacy.
    """
    processed: Doc = get_processed(tok_seq) if not spacy_processed_doc else spacy_processed_doc
    entities = []

    for ent in processed.ents:
        exact_match = tok_seq.get_word_token_span_from_chars(ent.start_char, ent.end_char, subtokens=True)
        entity = EntitySpan(tok_seq, exact_match)
        entities.append(entity)

    return entities


def get_spacy_sentences(tok_seq, spacy_processed_doc=None):
    processed: Doc = get_processed(tok_seq) if not spacy_processed_doc else spacy_processed_doc
    spacy_sents = processed.sents
    sentences = []
    from Code.Config import graph_construction_config

    for sent in spacy_sents:
        exact_match = tok_seq.get_word_token_span_from_chars(sent.start_char, sent.end_char, subtokens=True)
        sentence = DocumentExtract(tok_seq, exact_match, Code.constants.SENTENCE)
        sentences.append(sentence)

    return sentences
