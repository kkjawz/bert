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
import math

import json
import random
from tqdm import tqdm

import tokenization
import tensorflow as tf
import numpy as np

from coref.data import process_example

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("max_seq_length", 256, "Maximum sequence length.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 64,
    "Number of duplicate examples will be len(tokens) // dupe_factor.")


def write_examples_to_example_file(examples, tokenizer, max_seq_length, output_file):
    """Create TF example files from `TrainingInstance`s."""
    writer = tf.python_io.TFRecordWriter(output_file)

    total_written = 0
    for inst_index, example in enumerate(tqdm(examples, 'Writing Examples')):
        input_ids = tokenizer.convert_tokens_to_ids(example.tokens)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)

        input_mask = np.zeros(max_seq_length, np.int32)
        input_mask[:len(input_ids)] = 1
        segment_ids = [0] * max_seq_length

        speaker_ids = np.zeros(max_seq_length, np.int32)
        speaker_ids[:len(example.speaker_ids)] = example.speaker_ids

        mention_starts = [0] * max_seq_length
        mention_ends_ids = [-1] * max_seq_length
        for gold_start, gold_end in zip(example.gold_starts, example.gold_ends):
            mention_starts[gold_start] = 1
            mention_ends_ids[gold_start] = gold_end

        # clusters_mask(i, j) = 1 iff words i and j are on the same coreference cluster
        mention_clusters = np.zeros([max_seq_length, max_seq_length], np.int32)
        for i in range(len(example.gold_starts)):
            for j in range(len(example.gold_starts)):
                mention_clusters[example.gold_starts[i], example.gold_starts[j]] = example.cluster_ids[i] == example.cluster_ids[j]

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["speaker_ids"] = create_int_feature(speaker_ids)
        features["genre"] = create_int_feature([example.genre])
        features["mention_starts"] = create_int_feature(mention_starts)
        features["mention_ends_ids"] = create_int_feature(mention_ends_ids)
        features["mention_clusters"] = create_int_feature(mention_clusters.flatten())
        features["document_index"] = create_int_feature([example.document_index])
        features["document_offset"] = create_int_feature([example.offset])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writer.write(tf_example.SerializeToString())

        total_written += 1

        # if inst_index < 20:
        #     tf.logging.info("*** Example ***")
        #     tf.logging.info("tokens: %s" % " ".join(
        #         [tokenization.printable_text(x) for x in example.tokens]))
        #
        #     for feature_name in features.keys():
        #         feature = features[feature_name]
        #         values = []
        #         if feature.int64_list.value:
        #             values = feature.int64_list.value
        #         elif feature.float_list.value:
        #             values = feature.float_list.value
        #         tf.logging.info(
        #             "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    writer.close()
    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_examples(input_file, tokenizer, max_seq_length, dupe_factor, rng):
    """Create `TrainingInstance`s from raw text."""
    examples = []
    with open(input_file) as f:
        json_examples = [json.loads(jsonline) for jsonline in f.readlines()]

    for i, json_e in enumerate(tqdm(json_examples, desc='Creating Examples')):
        example = process_example(json_e, i, should_filter_embedded_mentions=True).bertify(tokenizer)

        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = max_seq_length - 3

        if len(example.tokens) > max_num_tokens:
            n_duplicates = math.ceil((len(example.tokens) - max_num_tokens) / dupe_factor) + 1
            for i in range(n_duplicates):
                # truncate tokens
                # start = rng.randint(0, len(example.tokens) - max_num_tokens)
                start = i * dupe_factor
                examples.append(example.truncate(start, max_num_tokens))
        else:
            examples.append(example)

    return examples


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=False)

    input_files = FLAGS.input_file.split(",")
    output_files = FLAGS.output_file.split(",")

    assert len(input_files) == len(output_files)

    tf.logging.info("*** Processing ***")
    rng = random.Random(FLAGS.random_seed)
    for input_file, output_file in tqdm(list(zip(input_files, output_files))):
        tf.logging.info("  input: %s", input_file)
        tf.logging.info("  output: %s", output_file)
        examples = create_examples(input_file, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor, rng)
        write_examples_to_example_file(examples, tokenizer, FLAGS.max_seq_length, output_file)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
