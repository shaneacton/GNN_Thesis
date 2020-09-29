from typing import List

from spacy.tokens.doc import Doc
import en_core_web_sm

from Code.Config import graph_construction_config
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan

nlp = en_core_web_sm.load()
loaded_neuralcoref = False


def get_spacy_entities(tok_seq, spacy_processed_doc=None) -> List[EntitySpan]:
    """
    uses the unprocessed text to perform NER using Spacy.
    """
    processed: Doc = nlp(tok_seq.text_obj.raw_text) if not spacy_processed_doc else spacy_processed_doc
    entities = []

    for ent in processed.ents:
        exact_match = tok_seq.get_word_token_span_from_chars(ent.start_char, ent.end_char, subtokens=True)
        entity = EntitySpan(tok_seq, exact_match)
        entities.append(entity)

    return entities


def get_spacy_sentences(tok_seq, spacy_processed_doc=None):
    processed: Doc = nlp(tok_seq.text_obj.raw_text) if not spacy_processed_doc else spacy_processed_doc
    spacy_sents = processed.sents
    sentences = []

    for sent in spacy_sents:
        exact_match = tok_seq.get_word_token_span_from_chars(sent.start_char, sent.end_char, subtokens=True)
        sentence = DocumentExtract(tok_seq, exact_match, graph_construction_config.SENTENCE)
        sentences.append(sentence)

    return sentences
