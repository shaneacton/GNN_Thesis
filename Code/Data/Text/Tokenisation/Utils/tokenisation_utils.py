from typing import List, Tuple, Dict

import Code.constants
from Code.Data.Text.Tokenisation.Utils import spacy_utils
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Data.Text.Tokenisation.entity_span import EntitySpan

USE_NEURAL_COREF = False
if USE_NEURAL_COREF:
    from Code.Data.Text.Tokenisation.Utils import neuralcoref_utils


def get_passages(tok_seq) -> List[DocumentExtract]:
    from Code.Data.Text.context import Context

    text_passages = tok_seq.text_obj.raw_text.split(Context.PASSAGE_BREAK_STRING)
    passages = []
    for text in text_passages:
        matches = find_string_in_subtokens(tok_seq, text)
        if len(matches) > 1:
            raise Exception("duplicate passage in token seq")

        # +1 to include one of the passage sep tokens for alignment reasons
        match = (matches[0][0], min(matches[0][1] + 1, len(tok_seq.raw_subtokens)))
        passage = DocumentExtract(tok_seq, match, Code.constants.PARAGRAPH)
        passages.append(passage)

    return passages


def get_sentences(tok_seq) -> List[DocumentExtract]:
    return spacy_utils.get_spacy_sentences(tok_seq)


def get_entities(tok_seq) -> List[EntitySpan]:
    return spacy_utils.get_spacy_entities(tok_seq)


def get_coreferences(tok_seq, entities) -> Dict[EntitySpan, List[EntitySpan]]:
    if USE_NEURAL_COREF:
        return neuralcoref_utils.get_neuralcoref_coreferences(tok_seq, entities)
    else:
        raise NotImplementedError()


def find_seq_in_seq(seq: List[str], query: List[str]) -> List[Tuple[int,int]]:
    """
    finds all the instances of the query sequence in the given sequence
    :param seq: a large token or subtoken sequence
    :param query: a smaller sequence which is being searched for in the larger sequence
    :return: all the (start,end ids) of the query which were found in the main seq
    """
    seq_index = 0
    num_seq_tokens = len(seq)
    num_query_tokens = len(query)
    matches = []  # list of tuples [start,end) ids of the matching subtokens

    def does_match_from(start_id):
        for i in range(1, num_query_tokens):
            if query[i] != seq[start_id + i]:
                return False
        return True

    while seq_index < num_seq_tokens:
        try:
            next_match_id = seq.index(query[0], seq_index, num_seq_tokens + 1)
            if does_match_from(next_match_id):
                matches.append((next_match_id, next_match_id + num_query_tokens))
            seq_index = next_match_id + 1
        except:  # no more matches
            break

    return matches


def find_string_in_subtokens(tok_seq, string):
    """
    tokenises the given string, and searches for matches in the subtoken sequence
    :returns the (start,end ids) of the string in the subtok seq
    """
    return find_seq_in_seq(tok_seq.raw_subtokens, tokeniser(string))
