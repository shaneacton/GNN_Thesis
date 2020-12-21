import re
import string
from collections import Counter
from typing import List

import nlp
import torch
from torch.utils.data import DataLoader
from transformers import LongformerForQuestionAnswering

from Code.Data.Text.text_utils import candidates, question
from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection
from Code.Models.GNNs.OutputModules.span_selection import SpanSelection
from Code.Training.Utils.initialiser import get_tokenizer
from Code.Training.Utils.text_encoder import TextEncoder
from Code.Training.Utils.dataset_utils import load_unprocessed_dataset


def normalize_answer(s: str):
    """
        from patil-suraj : longformer_qa_training
        Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth:str):
    """
        from patil-suraj : longformer_qa_training

    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str):
    """
        from patil-suraj : longformer_qa_training
    """
    em = (normalize_answer(prediction) == normalize_answer(ground_truth))
    return em


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    """
        from patil-suraj : longformer_qa_training
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_acc_and_f1(gold_answers: List[List[str]], predictions: List[str]):
    """
        from patil-suraj : longformer_qa_training
    """
    f1 = exact_match = total = 0

    # print("g:", len(gold_answers), "p:", len(predictions))
    # print(gold_answers)
    # print(predictions)
    for ground_truths, prediction in zip(gold_answers, predictions):
        if not prediction:
            continue
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def evaluate_full_gat(dataset_name, version_name, model, processed_valid_dataset):
    """
        evaluates a gat with a custom node selection based output
        input to these models is the example style input dict
    """
    print("evaluating model on", dataset_name, version_name)
    model = model.cuda()
    model.eval()

    dataloader = DataLoader(processed_valid_dataset, batch_size=1)
    tokenizer = get_tokenizer()

    encoder = TextEncoder(get_tokenizer())

    predictions = []
    with torch.no_grad():
        for batch in nlp.tqdm(dataloader):
            start_scores = None
            if not hasattr(model, "output_model") or isinstance(model.output_model, LongformerForQuestionAnswering):
                """models which do not have a dedicated output modules are assumed to be span prediction"""
                _, start_scores, end_scores = model(batch, return_dict=False)
            elif isinstance(model.output_model, SpanSelection):
                _, start_scores, end_scores = model(batch)

            elif isinstance(model.output_model, CandidateSelection):
                _, probs = model(batch)
                # print("probs:", probs, "cands:", candidates(batch), "ans:", batch['answer'], "q:", question(batch))
            else:
                raise Exception("unsupported output model: " + repr(model.output_model))

            qa = encoder.get_encoding(batch)
            if start_scores is not None:
                if torch.sum(start_scores) == 0:
                    """null output due to too large of an input"""
                    predictions.append(None)
                    continue
                # print("start scores 0:", start_scores.size(), start_scores)
                # print("end scores 0:", end_scores[0].size(), end_scores[0])
                s, e = torch.argmax(start_scores[0]), torch.argmax(end_scores[0]) + 1
                tokens = qa.tokens()[s: e]
                ans_ids = tokenizer.convert_tokens_to_ids(tokens)
                predicted = tokenizer.decode(ans_ids)
                # print("(s,e):", (s, e), "pred:", predicted, "total tokens:", len(qa.tokens()), "\n\n")
            else:
                if torch.sum(probs) == 0:
                    """null output due to too large of an input"""
                    predictions.append(None)
                    continue
                c = torch.argmax(probs[0])
                predicted = candidates(batch).split('</s>')[c]
                # print("predicted:", predicted)

            predictions.append(predicted)

    _compare_predictions(dataset_name, version_name, predictions)


def evaluate_span_model(dataset_name, version_name, model, processed_valid_dataset):
    """evaluates  token style models which input input_ids"""
    print("evaluating model on", dataset_name, version_name)
    model = model.cuda()
    model.eval()

    dataloader = DataLoader(processed_valid_dataset, batch_size=1)

    tokenizer = get_tokenizer()

    predictions = []
    with torch.no_grad():
        for batch in nlp.tqdm(dataloader):
            # print("model:", model)
            start_scores, end_scores = model(input_ids=batch['input_ids'].cuda(),
                                             attention_mask=batch['attention_mask'].cuda(), return_dict=False)
            # print("batch:", batch, "\nstart probs:", start_scores, "\n:end probs:", end_scores)
            if torch.sum(start_scores) == 0:
                """null output due to too large of an input"""
                predictions.append(None)
                continue

            all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
            s, e = torch.argmax(start_scores[0]), torch.argmax(end_scores[0]) + 1
            predicted = ' '.join(all_tokens[s: e])
            ans_ids = tokenizer.convert_tokens_to_ids(predicted.split())
            predicted = tokenizer.decode(ans_ids)
            predictions.append(predicted)

    _compare_predictions(dataset_name, version_name, predictions)


def _compare_predictions(dataset_name, version_name, predictions):
    gold_answers = []
    unprocessed_valid_dataset = load_unprocessed_dataset(dataset_name, version_name, nlp.Split.VALIDATION)

    for ref in unprocessed_valid_dataset:
        # print("ref:", ref)
        if 'answers' in ref:
            gold_answers.append(ref['answers']['text'])
        elif 'answer' in ref:
            gold_answers.append(ref['answer'])

    print(get_acc_and_f1(gold_answers, predictions))