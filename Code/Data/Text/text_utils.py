import copy
from typing import List, Union

from transformers import BatchEncoding


def words(encoding: BatchEncoding, query: str, context):
    words = []
    last_word_id = None
    last_chars = None
    text = query

    for (id, token, word_id) in zip(encoding["input_ids"], encoding.tokens(), encoding.words()):
        if id < 3:
            # special token
            words.append(token)
        elif last_word_id == word_id:
            # same word
            continue
        else:
            last_word_id = word_id
            chars = encoding.word_to_chars(word_id)
            if last_chars and chars.start < last_chars.end:
                # has switched text sources
                text = context
            last_chars = chars
            word = text[chars.start: chars.end]
            # print("word id:", word_id, "chars:", chars, "word:", word)
            words.append(word)

    return words


def is_batched(example):
    qkey = question_key(example)
    return isinstance(example[qkey], List)


def get_example_i_from_batch(example, i):
    """takes in a batched example with batch size = n and returns a single version"""
    if not is_batched(example):
        raise Exception()
    single = copy.deepcopy(example)
    if candidates(single):
        single['candidates'] = candidates(example)[i]
    single[context_key(example)] = context(example)[i]
    single[question_key(example)] = question(example)[i]
    return single


def get_single_example(example):
    """takes in a batched example with batch size = 1 and returns a single version"""
    single = get_example_i_from_batch(example, 0)
    bs = len(context(example))
    if bs > 1:
        raise Exception("cannot get single example from batch with size = " + repr(bs) + " >1")
    return single


def get_single_value(example, key):
    if isinstance(example[key], List):
        raise Exception("cannot get " + key + " from ex, it appears to be batched: " + repr(example))
    return example[key]


def question_key(example):
    if 'question' in example:
        key = 'question'
    elif 'query' in example:
        key = 'query'
    else:
        raise Exception("can't get query from " + repr(example))
    return key


def question(example) -> Union[str, List[str]]:
    if is_batched(example):
        return example[question_key(example)]
    return get_single_value(example, question_key(example))


def context_key(example):
    if 'context' in example:
        return 'context'
    elif 'supports' in example:
        return 'supports'
    raise Exception()


def context(example) -> Union[str, List[str]]:
    if 'context' in example:
        return example['context']
    elif 'supports' in example:
        # multiple passages for a single context, must combine
        return " ".join(example['supports'])
    raise Exception("can't get context from " + repr(example))


def candidates(example):
    """not all examples have candidates"""
    if not has_candidates(example):
        return None

    cands = example["candidates"]
    return cands


def num_candidates(example):
    cands = candidates(example)
    if isinstance(cands, str):
        return len(cands.split("</s>"))  # todo support for different sep toks
    return len(cands)


def has_candidates(example):
    return 'candidates' in example
