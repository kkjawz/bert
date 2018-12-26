# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 1,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_position, masked_lm_label, masked_lm_label_pos,
                 masked_lm_candidates, masked_lm_candidates_pos):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_position = masked_lm_position
        self.masked_lm_label = masked_lm_label
        self.masked_lm_label_pos = masked_lm_label_pos
        self.masked_lm_candidates = masked_lm_candidates
        self.masked_lm_candidates_pos = masked_lm_candidates_pos

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "masked_lm_position: %s\n" % (str(self.masked_lm_position))
        s += "masked_lm_label: %s\n" % (tokenization.printable_text(self.masked_lm_label))
        s += "masked_lm_label_pos: {}\n".format(self.masked_lm_label_pos)
        s += "masked_lm_candidates: %s\n" % (" ".join(
            [tokenization.printable_text(self.masked_lm_candidates) for x in self.masked_lm_candidates]))
        s += "masked_lm_candidates_pos: {}\n".format(self.masked_lm_candidates_pos)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(all_instances, tokenizer, max_seq_length, cbtest_vocab, output_files):
    """Create TF example files from `TrainingInstance`s."""
    print(len(all_instances))
    print(output_files)
    assert len(all_instances) == len(output_files)
    cbtest_stoi = {s: i for i, s in enumerate(cbtest_vocab)}

    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    total_written = 0
    for i, instances in enumerate(all_instances):
        for (inst_index, instance) in enumerate(instances):
            input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = list(instance.segment_ids)
            assert len(input_ids) <= max_seq_length

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            masked_lm_positions = [instance.masked_lm_position]
            masked_lm_ids = [cbtest_stoi[instance.masked_lm_label]]
            masked_lm_candidate_ids = [instance.masked_lm_candidates.index(instance.masked_lm_label)]
            masked_lm_weights = [1.0]
            masked_lm_candidates = [cbtest_stoi[c] for c in instance.masked_lm_candidates]

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(input_ids)
            features["input_mask"] = create_int_feature(input_mask)
            features["segment_ids"] = create_int_feature(segment_ids)
            features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
            features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
            features["masked_lm_candidate_ids"] = create_int_feature(masked_lm_candidate_ids)
            features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
            features["masked_lm_candidates"] = create_int_feature(masked_lm_candidates)
            features["masked_lm_label_pos"] = create_int_feature([instance.masked_lm_label_pos])
            features["masked_lm_candidates_pos"] = create_int_feature(instance.masked_lm_candidates_pos)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            writers[i].write(tf_example.SerializeToString())

            total_written += 1

            if inst_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in instance.tokens]))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    values = []
                    if feature.int64_list.value:
                        values = feature.int64_list.value
                    elif feature.float_list.value:
                        values = feature.float_list.value
                    tf.logging.info(
                        "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


class RawExample(object):
    def __init__(self):
        self.story = []
        self.label = None
        self.label_pos = None
        self.candidates = None
        self.candidates_pos = None


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_raw_examples = []
    cbtest_vocab = collections.Counter()

    for input_file in input_files:
        cur_raw_examples = []
        cur_example = RawExample()
        story_orig_tokens = []
        story_orig_to_tok_map = []
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    cur_example = RawExample()
                    story_orig_tokens = []
                    story_orig_to_tok_map = []
                    continue

                line_index, line = line.split(' ', 1)

                if int(line_index) == 21:
                    # Remove any leading or trailing whitespace after splitting
                    line, label, _, candidates_string = (x.strip() for x in line.split('\t'))
                    orig_tokens = line.split()
                    story_orig_tokens.extend(orig_tokens)
                    candidates = [c for c in candidates_string.split('|') if c]

                    if len(candidates) < 10:
                        print('BAD CANDIDATES: ', candidates_string)
                        cur_example = RawExample()
                        story_orig_tokens = []
                        story_orig_to_tok_map = []
                        continue

                    for orig_token in orig_tokens:
                        story_orig_to_tok_map.append(len(cur_example.story))
                        if orig_token == 'XXXXX':
                            cur_example.story.append('[MASK]')
                        else:
                            cur_example.story.extend(tokenizer.tokenize(orig_token))

                    label_orig_index = len(story_orig_tokens) - 1 - story_orig_tokens[::-1].index(label)
                    label_index = story_orig_to_tok_map[label_orig_index]

                    candidates_index = []
                    for c in candidates:
                        c_orig_index = len(story_orig_tokens) - 1 - story_orig_tokens[::-1].index(c)
                        candidates_index.append(story_orig_to_tok_map[c_orig_index])

                    cur_example.label = label.lower()
                    cur_example.label_pos = label_index
                    cur_example.candidates = [c.lower() for c in candidates]
                    cur_example.candidates_pos = candidates_index

                    cur_raw_examples.append(cur_example)
                else:
                    orig_tokens = line.split()
                    story_orig_tokens.extend(orig_tokens)
                    for orig_token in orig_tokens:
                        story_orig_to_tok_map.append(len(cur_example.story))
                        cur_example.story.extend(tokenizer.tokenize(orig_token))

                cbtest_vocab.update(line.lower().split())

        # Remove empty documents
        cur_raw_examples = [e for e in cur_raw_examples if e.story]
        rng.shuffle(cur_raw_examples)

        all_raw_examples.append(cur_raw_examples)

    vocab_words = list(tokenizer.vocab.keys())

    all_instances = []
    for cur_raw_examples in all_raw_examples:
        cur_instances = []
        for document_index in range(len(cur_raw_examples)):
            instance = create_instance_from_example(
                    cur_raw_examples, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
            if instance is not None:
                cur_instances.append(instance)
            else:
                print('Whoops!')

        rng.shuffle(cur_instances)
        all_instances.append(cur_instances)

    cbtest_vocab = list(zip(*cbtest_vocab.most_common()))[0]

    return all_instances, cbtest_vocab


def create_instance_from_example(
        all_examples, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    example = all_examples[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # truncate beginning of story
    tokens = example.story[-max_num_tokens:]
    n_truncated = len(example.story) - len(tokens)
    segment_ids = [0] * len(tokens)
    masked_lm_position = tokens.index('[MASK]')
    masked_lm_label = example.label
    masked_lm_label_pos = example.label_pos - n_truncated
    masked_lm_candidates = example.candidates
    masked_lm_candidates_pos = [c - n_truncated for c in example.candidates_pos]

    if masked_lm_label_pos < 0 or any([c < 0 for c in masked_lm_candidates_pos]):
        return None

    instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        masked_lm_position=masked_lm_position,
        masked_lm_label=masked_lm_label,
        masked_lm_label_pos=masked_lm_label_pos,
        masked_lm_candidates=masked_lm_candidates,
        masked_lm_candidates_pos=masked_lm_candidates_pos)

    return instance


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    instances, cbtest_vocab = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, cbtest_vocab, output_files)

    with tf.gfile.GFile(FLAGS.output_vocab_file, mode='w') as output_vocab_file:
        for w in cbtest_vocab:
            output_vocab_file.write(w + '\n')



if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("output_vocab_file")
    tf.app.run()
