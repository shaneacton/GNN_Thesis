from typing import List, Dict

import neuralcoref
from neuralcoref.neuralcoref import Cluster
from spacy.tokens.doc import Doc

from Code.Data.Text.Tokenisation.Utils import spacy_utils
from Code.Data.Text.Tokenisation.Utils.spacy_utils import nlp
from Code.Data.Text.Tokenisation.entity_span import EntitySpan


def get_neuralcoref_coreferences(tok_seq, entities: List[EntitySpan], spacy_processed_doc=None) -> Dict[EntitySpan, List[EntitySpan]]:
    if not spacy_utils.loaded_neuralcoref:
        """must add neuralcoref and reprocess doc"""
        neuralcoref.add_to_pipe(nlp)
        spacy_processed_doc = None
        spacy_utils.loaded_neuralcoref = True

    processed: Doc = spacy_utils.get_processed(tok_seq) if not spacy_processed_doc else spacy_processed_doc
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