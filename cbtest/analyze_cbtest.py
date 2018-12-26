import collections
import tensorflow as tf

import tokenization


class RawExample(object):
    def __init__(self):
        self.story = []
        self.label = None
        self.candidates = None


def create_training_instances(input_files, tokenizer):
    """Create `TrainingInstance`s from raw text."""
    all_raw_examples = [RawExample()]
    cbtest_vocab = collections.Counter()

    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_raw_examples.append(RawExample())
                    continue

                line_index, line = line.split(' ', 1)

                if int(line_index) == 21:
                    # Remove any leading or trailing whitespace after splitting
                    line, label, _, candidates_string = (x.strip() for x in line.split('\t'))
                    label = label.lower()
                    candidates = [c.lower() for c in candidates_string.split('|') if c]

                    if len(candidates) < 10:
                        print('BAD CANDIDATES: ', candidates_string)
                        del all_raw_examples[-1]
                        continue

                    assert label.lower() in candidates

                    all_raw_examples[-1].label = label
                    all_raw_examples[-1].candidates = candidates

                    tokens = tokenizer.tokenize(line)
                else:
                    tokens = tokenizer.tokenize(line)

                if tokens:
                    all_raw_examples[-1].story.extend(tokens)

                cbtest_vocab.update(line.lower().split())

    all_raw_examples = [e for e in all_raw_examples if e.story]

    cbtest_vocab = list(zip(*cbtest_vocab.most_common()))[0]

    return all_raw_examples, cbtest_vocab


if __name__ == '__main__':
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    examples_train, vocab_train = create_training_instances(['/home/benk/data/CBTest/data/cbtest_NE_train.txt'], tokenizer)
    examples_test, vocab_test = create_training_instances(['/home/benk/data/CBTest/data/cbtest_NE_test_2500ex.txt'], tokenizer)
    vocab_train_set = set(vocab_train)

    stats = collections.defaultdict(lambda: 0)

    cands_not_in_vocab = 0
    for e in examples_train:
        for c in e.candidates:
            if c not in vocab_train:
                cands_not_in_vocab += 1
    print('cands_not_in_vocab:', cands_not_in_vocab)

    test_cands_not_in_vocab = 0
    test_labels_not_in_vocab = 0
    for e in examples_test:
        for c in e.candidates:
            if c not in vocab_train:
                test_cands_not_in_vocab += 1
                break
        if e.label not in vocab_train:
            test_labels_not_in_vocab += 1
    print('test_cands_not_in_vocab:', test_cands_not_in_vocab)
    print('test_cands_not_in_vocab_pct:', test_cands_not_in_vocab / len(examples_test))

    print('test_cands_not_in_vocab:', test_labels_not_in_vocab)
    print('test_labels_not_in_vocab_pct:', test_labels_not_in_vocab / len(examples_test))

    for e in examples_train:
        for c in e.candidates:
            if c not in e.story:
                stats['cands_not_in_story'] += 1
                break

        for c in e.candidates:
            if c not in vocab_train:
                stats['cands_not_in_vocab'] += 1
                break

        if e.label not in e.story:
            stats['labels_not_in_story'] += 1

    stats = collections.defaultdict(lambda: 0)
    for i, e in enumerate(examples_test):
        for c in e.candidates:
            if c not in e.story:
                stats['test_cands_not_in_story'] += 1
                break

        for c in e.candidates:
            if c not in vocab_train:
                stats['test_cands_not_in_vocab'] += 1
                break

        if e.label not in e.story:
            stats['test_labels_not_in_story'] += 1

        if e.label not in vocab_train:
            stats['test_labels_not_in_vocab'] += 1

    for s in stats:
        print(f'{s}: {stats[s]} ({stats[s] / len(examples_test)}%)')
