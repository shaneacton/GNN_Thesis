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


def question(example):
    if 'question' in example:
        return example['question']
    elif 'query' in example:
        return example['query']
    else:
        raise Exception("can't get query from " + repr(example))


def context(example):
    if 'context' in example:
        return example['context']
    else:
        raise Exception("can't get context from " + repr(example))
