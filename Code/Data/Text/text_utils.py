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


def get_single_value(example, key):
    if isinstance(example[key], List):
        raise Exception("cannot get " + key + " from ex, it appears to be batched: " + repr(example))
    return example[key]


def question_key(example):
    if 'question' in example:
        key = 'question'
    elif 'query' in example:
        key = example['query']
    else:
        raise Exception("can't get query from " + repr(example))
    return key


def question(example) -> Union[str, List[str]]:
    if is_batched(example):
        return example[question_key(example)]
    return get_single_value(example, question_key(example))


def context(example) -> Union[str, List[str]]:
    if 'context' in example:
        context = example['context']
    elif 'supports' in example:
        # multiple passages for a single context, must combine
        raise NotImplementedError()
    else:
        raise Exception("can't get context from " + repr(example))
    return context