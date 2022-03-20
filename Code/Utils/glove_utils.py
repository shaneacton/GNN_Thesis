from typing import List, Tuple

from transformers import TokenSpan

from Code.Embedding.glove_embedder import GloveEmbedder


def get_glove_entity_token_spans(example, glove_embedder: GloveEmbedder, use_nouns=False) -> List[List[Tuple[int]]]:
    """returns a 2d list indexed [supp_id][ent_in_supp]"""
    all_token_spans: List[List[Tuple[int]]] = []
    for s, support in enumerate(example.supports):
        """get entity node embeddings"""
        doc_token_spans = get_special_entity_token_spans(example, support, glove_embedder)
        all_token_spans.append(doc_token_spans)
    return all_token_spans


def get_special_entity_token_spans(example, support, glove_embedder) -> List[Tuple[int]]:
    passage_words: List[str] = glove_embedder.get_words(support)
    subject_words: List[str] = glove_embedder.get_words(example.query_subject)
    candidate_words: List[List[str]] = [glove_embedder.get_words(cand) for cand in example.candidates]
    special_words: List[List[str]] = candidate_words + [subject_words]

    token_spans = []
    for specials in special_words:
        if len(specials) <= 0:
            continue
        start = specials[0]
        indices = [i for i, x in enumerate(passage_words) if x == start]
        for i in indices:
            corr_passage_words = passage_words[i:i+len(specials)]
            if corr_passage_words == specials:
                # print("found overlap! special:", specials, "subject:", subject_words, "cands:", candidate_words)
                token_spans.append(TokenSpan(i, i+len(specials)))
    return token_spans


class NoEntityTokens(Exception):
    pass