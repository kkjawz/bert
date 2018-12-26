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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle

import modeling
import optimization
import tensorflow as tf

import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "cbtest_vocab_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_pred", False, "Whether to run predict on the dev set.")

flags.DEFINE_bool("filter_oov", False, "Whether to filter examples that's answer is out of train's vocabulary.")

flags.DEFINE_bool("pos", False, "Set to true for positional training.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("vanilla_bert", False, "")


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, cbtest_vocab_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_candidate_ids = features["masked_lm_candidate_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        masked_lm_candidates = features["masked_lm_candidates"]
        masked_lm_label_pos = features["masked_lm_label_pos"]
        masked_lm_candidates_pos = features["masked_lm_candidates_pos"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if FLAGS.pos:
            (masked_lm_loss,
             masked_lm_example_loss, masked_lm_log_probs) = get_cbtest_positional_output(
                bert_config, model.get_sequence_output(),
                masked_lm_positions, masked_lm_candidate_ids, masked_lm_candidates_pos)
        else:
            (masked_lm_loss,
             masked_lm_example_loss, masked_lm_log_probs) = get_cbtest_output(
                bert_config, model.get_sequence_output(),
                masked_lm_positions, masked_lm_candidate_ids, masked_lm_weights, masked_lm_candidates, cbtest_vocab_size)

        total_loss = masked_lm_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                                 [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_candidate_ids,
                masked_lm_weights
            ])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                             [-1, masked_lm_log_probs.shape[-1]])
            masked_lm_predictions = tf.argmax(
                masked_lm_log_probs, axis=-1, output_type=tf.int32)

            predictions = {
                'log_probabilities': masked_lm_log_probs,
                'prediction': masked_lm_predictions,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'masked_lm_candidate_ids': masked_lm_candidate_ids,
                'masked_lm_candidates': masked_lm_candidates,
            }
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_cbtest_output(bert_config, input_tensor, positions, label_ids, label_weights, candidate_ids, cbtest_vocab_size):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cbtest/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        output_weights = tf.get_variable(
            "output_weights", [cbtest_vocab_size, bert_config.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [cbtest_vocab_size], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        candidate_log_probs = tf.batch_gather(log_probs, candidate_ids)

        label_ids = tf.reshape(label_ids, [-1])

        one_hot_labels = tf.one_hot(label_ids, depth=10, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(candidate_log_probs * one_hot_labels, axis=[-1])
        loss = tf.reduce_mean(per_example_loss)
        # numerator = tf.reduce_sum(label_weights * per_example_loss)
        # denominator = tf.reduce_sum(label_weights) + 1e-5
        # loss = numerator / denominator

    return (loss, per_example_loss, candidate_log_probs)


def get_cbtest_positional_output(bert_config, input_tensor, label_positions, label_candidate_index, candidate_positions):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    """Get loss and log probs for the masked LM."""
    with tf.variable_scope("cbtest/predictions"):
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length (= 1)
        #   T = `to_tensor` sequence length (= 10)
        #   N = `num_attention_heads` (= 1)
        #   H = `size_per_head` (= bert_config.hidden_size)

        batch_size, n_candidates = modeling.get_shape_list(candidate_positions, expected_rank=2)

        candidates_tensor = gather_indexes(input_tensor, candidate_positions)
        candidates_tensor = tf.reshape(candidates_tensor, [batch_size, n_candidates, bert_config.hidden_size])

        orig_shape = modeling.get_shape_list(candidates_tensor, expected_rank=[2, 3])
        num_attention_heads = 1

        if FLAGS.vanilla_bert:
            # `masked_tensor` = [B, F, H]
            query_layer = gather_indexes(input_tensor, label_positions)
            key_layer = candidates_tensor
        else:
            candidates_tensor = tf.layers.dense(
                candidates_tensor,
                bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                name="transform",
                kernel_initializer=modeling.create_initializer(0.02))

            # `masked_tensor` = [B, F, H]
            masked_tensor = gather_indexes(input_tensor, label_positions)
            masked_tensor = tf.expand_dims(masked_tensor, axis=[1])

            candidates_2d = modeling.reshape_to_matrix(candidates_tensor)

            # `query_layer` = [B*F, N*H]
            query_layer = tf.layers.dense(
                masked_tensor,
                bert_config.hidden_size * num_attention_heads,
                activation=None,
                name="query",
                kernel_initializer=modeling.create_initializer(0.02))

            # `key_layer` = [B*T, N*H]
            key_layer = tf.layers.dense(
                candidates_2d,
                bert_config.hidden_size * num_attention_heads,
                activation=None,
                name="key",
                kernel_initializer=modeling.create_initializer(0.02))

        # `query_layer` = [B, N, F, H]
        query_layer = transpose_for_scores(query_layer, batch_size,
                                           num_attention_heads, 1,
                                           bert_config.hidden_size)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                         orig_shape[1], bert_config.hidden_size)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        # `attention_scores` = [B, N, F, T]
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(bert_config.hidden_size)))

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = tf.nn.log_softmax(attention_scores)

        # `attention_probs` = [B, T]
        attention_probs = tf.squeeze(attention_probs, axis=[1, 2])

        label_candidate_index = tf.reshape(label_candidate_index, [-1])
        one_hot_labels = tf.one_hot(label_candidate_index, depth=10, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(attention_probs * one_hot_labels, axis=[-1])
        loss = tf.reduce_mean(per_example_loss)
        # numerator = tf.reduce_sum(label_weights * per_example_loss)
        # denominator = tf.reduce_sum(label_weights) + 1e-5
        # loss = numerator / denominator

    return loss, per_example_loss, attention_probs


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_candidate_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "masked_lm_candidates":
                tf.FixedLenFeature([10], tf.int64),
            "masked_lm_label_pos":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_candidates_pos":
                tf.FixedLenFeature([10], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)

            # We must `drop_remainder` on training because the TPU requires fixed
            # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
            # and we *don't* want to drop the remainder, otherwise we wont cover
            # every sample.
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=True))
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            # d = d.repeat()

            d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
            if FLAGS.filter_oov:
                with open(os.path.expandvars('$DATA/CBTest/data/out_of_vocab.pkl'), 'rb') as f:
                    out_of_vocab = pickle.load(f)
                    out_of_vocab = tf.constant(out_of_vocab)
                d = d.filter(lambda e: tf.logical_not(tf.reduce_any(tf.equal(out_of_vocab, e["masked_lm_ids"][0]))))
            # d = d.filter(lambda e: tf.reduce_any(tf.equal(out_of_vocab, e["masked_lm_ids"][0])))
            d = d.batch(batch_size=batch_size, drop_remainder=True)
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_pred:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_pred` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    with tf.gfile.GFile(FLAGS.cbtest_vocab_file) as cbtest_vocab_file:
        cbtest_vocab = [w for w in cbtest_vocab_file.read().split('\n') if w]
        cbtest_vocab_size = len(cbtest_vocab)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        cbtest_vocab_size=cbtest_vocab_size)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        train_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_pred:
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=True)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        predictions = estimator.predict(input_fn=eval_input_fn)

        for p in predictions:
            log_probabilities = p['log_probabilities']
            pred = p['prediction']
            gt = p['masked_lm_candidate_ids'][0]
            input_ids = p['input_ids']
            input_mask = p['input_mask']
            candidate_ids = p['masked_lm_candidates']

            if pred != gt:
                input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                input_tokens = [t for t, m in zip(input_tokens, input_mask) if m]
                story = ' '.join(input_tokens)
                story = story.split(' .')
                story = ' .\n'.join(story)

                print('story:', story)
                print('pred:', cbtest_vocab[candidate_ids[pred]])
                print('gt:', cbtest_vocab[candidate_ids[gt]])

                candidates = [cbtest_vocab[i] for i in candidate_ids]
                probabilities = [math.exp(l) for l in log_probabilities]
                candidate_print = ', '.join(f'{c}: {p:.2f}' for c, p in zip(candidates, probabilities))
                print('candidates:', candidate_print)
                print('')

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("cbtest_vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
