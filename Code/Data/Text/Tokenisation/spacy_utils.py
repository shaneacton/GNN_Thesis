from typing import List

from neuralcoref.neuralcoref import Cluster
from spacy.tokens.doc import Doc
import en_core_web_sm
import neuralcoref

from Code.Config import configuration
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan

nlp = en_core_web_sm.load()
neuralcoref.add_to_pipe(nlp)


def get_spacy_entities(tok_seq, spacy_processed_doc=None) -> List[EntitySpan]:
    """
    uses the unprocessed text to perform NER using Spacy.
    """
    processed: Doc = nlp(tok_seq.text_obj.raw_text) if not spacy_processed_doc else spacy_processed_doc
    entities = []

    for ent in processed.ents:
        exact_match = tok_seq.get_word_token_span_from_chars(ent.start_char, ent.end_char, subtokens=True)
        entity = EntitySpan(tok_seq, exact_match, ent)
        entities.append(entity)

    return entities


def get_spacy_coreferences(tok_seq, entities, spacy_processed_doc=None):
    processed: Doc = nlp(tok_seq.text_obj.raw_text) if not spacy_processed_doc else spacy_processed_doc
    clusters: List[Cluster] = processed._.coref_clusters

    corefs = {}
    for cluster in clusters:
        token_matches = [tok_seq.get_word_token_span_from_chars(mention.start_char, mention.end_char, subtokens=True)
                         for mention in cluster.mentions]
        entities = [EntitySpan(tok_seq, token_matches[i], cluster.mentions[i], is_coref=i > 0)
                    for i in range(len(token_matches))]
        corefs[entities[0]] = entities[1:]  # ent[0] is main, rest are corefs

    corefs = {ent: corefs[ent] for ent in entities if ent in corefs}
    # replaces the main ents produced by the ents produced by NER as these have more metadata
    return corefs


def get_spacy_sentences(tok_seq, spacy_processed_doc=None):
    processed: Doc = nlp(tok_seq.text_obj.raw_text) if not spacy_processed_doc else spacy_processed_doc
    spacy_sents = processed.sents
    sentences = []

    for sent in spacy_sents:
        exact_match = tok_seq.get_word_token_span_from_chars(sent.start_char, sent.end_char, subtokens=True)
        sentence = DocumentExtract(tok_seq, exact_match, configuration.SENTENCE, sent)
        sentences.append(sentence)

    return sentences
