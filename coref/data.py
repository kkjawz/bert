import numpy as np
import random

import tokenization


class Example(object):
    def __init__(self, tokens, gold_starts, gold_ends, speaker_ids, cluster_ids):
        self.tokens = tokens
        self.gold_starts = gold_starts
        self.gold_ends = gold_ends
        self.speaker_ids = speaker_ids
        self.cluster_ids = cluster_ids

    def truncate(self, start, size):
        # don't truncate in the middle of a mention
        for mention in zip(self.gold_starts, self.gold_ends):
            if index_in_mention(start, mention):
                start = mention[0]

            if index_in_mention(start + size, mention):
                size -= start + size - mention[0]
        end = start + size

        tokens = self.tokens[start:end]
        speaker_ids = self.speaker_ids[start:end]
        gold_spans = np.logical_and(self.gold_starts >= start, self.gold_ends < end)
        gold_starts = self.gold_starts[gold_spans] - start
        gold_ends = self.gold_ends[gold_spans] - start
        cluster_ids = self.cluster_ids[gold_spans]

        return Example(tokens, gold_starts, gold_ends, speaker_ids, cluster_ids)

    def bertify(self, tokenizer):
        bert_tokens = []
        orig_to_bert_map = []
        for t in self.tokens:
            orig_to_bert_map.append(len(bert_tokens))
            bert_tokens.extend(tokenizer.tokenize(t))

        orig_to_bert_map = np.array(orig_to_bert_map)
        if len(self.gold_starts):
            gold_starts = orig_to_bert_map[self.gold_starts]
            gold_ends = orig_to_bert_map[self.gold_ends]
        else:
            gold_starts = self.gold_starts
            gold_ends = self.gold_ends

        return Example(bert_tokens, gold_starts, gold_ends, self.speaker_ids, self.cluster_ids)


def index_in_mention(index, mention):
    return mention[0] <= index and mention[1] >= index


def mention_contains(mention1, mention2):
    return mention1[0] <= mention2[0] and mention1[1] >= mention2[1]


def filter_embedded_mentions(mentions):
    """
    Filter out mentions embedded in other mentions
    """
    filtered = []
    for i, m in enumerate(mentions):
        other_mentions = mentions[:i] + mentions[i + 1:]
        if any(mention_contains(other_m, m) for other_m in other_mentions):
            continue
        filtered.append(m)
    return filtered


def flatten(l):
    return [item for sublist in l for item in sublist]


def tensorize_mentions(mentions):
    if len(mentions) > 0:
        starts, ends = zip(*mentions)
    else:
        starts, ends = [], []
    return np.array(starts), np.array(ends)


def process_example(example, should_filter_embedded_mentions=False):
    clusters = example["clusters"]

    gold_mentions = sorted(tuple(m) for m in flatten(clusters))
    if should_filter_embedded_mentions:
        gold_mentions = filter_embedded_mentions(gold_mentions)
    gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
    cluster_ids = np.zeros(len(gold_mentions))
    for cluster_id, cluster in enumerate(clusters):
        for mention in cluster:
            if tuple(mention) in gold_mention_map:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)
    speakers = flatten(example["speakers"])

    assert num_words == len(speakers)

    all_text = ' '.join(' '.join(s) for s in sentences)
    all_text = tokenization.convert_to_unicode(all_text)
    tokens = all_text.split(' ')

    speaker_dict = {s: i for i, s in enumerate(set(speakers))}
    speaker_ids = np.array([speaker_dict[s] for s in speakers])

    # TODO: genre
    # doc_key = example["doc_key"]
    # genre = self.genres[doc_key[:2]]

    gold_starts, gold_ends = tensorize_mentions(gold_mentions)

    return Example(tokens, gold_starts, gold_ends, speaker_ids, cluster_ids)
