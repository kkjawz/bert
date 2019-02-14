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
from collections import OrderedDict, defaultdict

import json
import numpy as np
import os
import tensorflow as tf
from colorama import Fore, Back, Style
from tensorflow.python.estimator.estimator import _write_dict_to_summary

import modeling
import optimization
import tokenization
from coref import metrics
from coref.chinese_whispers import chinese_whispers
from coref.data import process_example
from modeling import reshape_to_matrix, attention_scores_layer, get_shape_list, gelu, create_initializer
import unionfind

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
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_pred", False, "Whether to run predict on the dev set.")

flags.DEFINE_bool("print_results", False, "")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 15000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 100, "Number of warmup steps.")

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


flags.DEFINE_bool("add_cluster_features", False, "")
flags.DEFINE_string("cluster_loss_type", None, "")


def create_mention_ends_mask(batch_size, seq_len=None):
    if seq_len is None:
        seq_len = FLAGS.max_seq_length
    mention_ends_mask = np.zeros([seq_len, seq_len], np.int32)
    for i in range(seq_len):
        mention_ends_mask[i, i:] = 1
    mention_ends_mask = np.tile(mention_ends_mask[None, :, :], [batch_size, 1, 1])
    return mention_ends_mask


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        speaker_ids = features["speaker_ids"]
        genre = features["genre"]
        mention_starts = features["mention_starts"]
        mention_ends_ids = features["mention_ends_ids"]
        mention_clusters = features["mention_clusters"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        mention_ends_mask = create_mention_ends_mask(FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size)
        mention_ends_mask = tf.constant(mention_ends_mask, name='mention_ends_mask')

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        coref_outputs = get_coref_outputs(
            bert_config=bert_config,
            input_tensor=model.get_sequence_output(),
            input_mask=input_mask,
            speaker_ids=speaker_ids,
            genre=genre,
            mention_starts=mention_starts,
            mention_ends_ids=mention_ends_ids,
            mention_ends_mask=mention_ends_mask,
            mention_clusters=mention_clusters)

        coref_metrics = get_coref_metrics(
            coref_outputs=coref_outputs,
            input_mask=input_mask,
            mention_starts=mention_starts,
            mention_ends_ids=mention_ends_ids,
            mention_clusters=mention_clusters,
            mention_ends_mask=mention_ends_mask)

        total_loss = coref_outputs['loss']

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

            logging_hook = tf.train.LoggingTensorHook({'l': coref_outputs['loss'],
                                                       'sl': coref_outputs['mention_starts_loss'],
                                                       'el': coref_outputs['mention_ends_loss'],
                                                       'cl': coref_outputs['mention_clusters_loss'],
                                                       'sr': coref_metrics['mention_starts_recall'][1],
                                                       'sp': coref_metrics['mention_starts_precision'][1],
                                                       'ea': coref_metrics['mention_ends_accuracy'][1],
                                                       'cr': coref_metrics['mention_clusters_recall'][1],
                                                       'cp': coref_metrics['mention_clusters_precision'][1]},
                                                      every_n_secs=10)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metric_ops = {'Average F1 (py)': (f, tf.no_op()),
                               'Average precision (py)': (p, tf.no_op()),
                               'Average recall (py)': (r, tf.no_op()),
                               'mention_starts_recall': coref_metrics['mention_starts_recall'],
                               'mention_starts_precision': coref_metrics['mention_starts_precision'],
                               'mention_ends_accuracy': coref_metrics['mention_ends_accuracy'],
                               'mention_clusters_recall': coref_metrics['mention_clusters_recall'],
                               'mention_clusters_precision': coref_metrics['mention_clusters_precision']}

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=(eval_fn, [coref_outputs['mention_starts_scores'],
                                        coref_outputs['mention_ends_scores'],
                                        coref_outputs['mention_clusters_scores'],
                                        input_mask,
                                        mention_ends_ids,
                                        mention_clusters]),
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            metric_to_batch = lambda x: tf.tile(tf.reshape(x[1], [1, 1]), [FLAGS.eval_batch_size, 1])
            predictions = {
                'pred_start_scores': coref_outputs['mention_starts_scores'],
                'pred_end_scores': coref_outputs['mention_ends_scores'],
                'pred_end_query': coref_outputs['mention_ends_query'],
                'pred_end_key': coref_outputs['mention_ends_key'],
                'pred_cluster_scores': coref_outputs['mention_clusters_scores'],
                'pred_cluster_features': coref_outputs['mention_clusters_features'],
                'input_mask': input_mask,
                'gold_end_ids': mention_ends_ids,
                'gold_clusters': mention_clusters,
                'input_ids': input_ids,
                'mention_starts_recall': metric_to_batch(coref_metrics['mention_starts_recall']),
                'mention_starts_precision': metric_to_batch(coref_metrics['mention_starts_precision']),
                'mention_ends_accuracy': metric_to_batch(coref_metrics['mention_ends_accuracy']),
                'mention_clusters_recall': metric_to_batch(coref_metrics['mention_clusters_recall']),
                'mention_clusters_precision': metric_to_batch(coref_metrics['mention_clusters_precision']),
                'document_index': features["document_index"],
                'document_offset': features["document_offset"],
            }
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def bucket_distance(distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)


def get_mention_width_feature(end_scores, batch_size, seq_length):
    i0 = []
    i1 = []
    i2 = []
    mask\
        = np.zeros([batch_size, seq_length, 31], np.bool)
    for b in range(batch_size):
        for i in range(seq_length):
            for j in range(i, i + 31):
                i0.append(b)
                i1.append(i)
                i2.append(min(j, seq_length - 1))

                if j < seq_length:
                    mask[b, i, j - i] = 1
    gather_indices = tf.constant(np.array(np.stack([i0, i1, i2], axis=-1).reshape(batch_size, seq_length, 31, 3)))
    mask = tf.constant(mask)
    mention_width_features = tf.gather_nd(end_scores, gather_indices)
    mention_width_features = tf.reshape(mention_width_features, [batch_size, seq_length, 31])
    mention_width_features = mention_width_features * tf.to_float(mask)
    return mention_width_features


def get_mention_distances_feature(batch_size, seq_length):
    ranges = np.tile(np.arange(seq_length)[None, :], [seq_length, 1])
    distances = abs(ranges - np.arange(seq_length)[:, None])
    distances = bucket_distance(distances)
    distances = tf.tile(distances[None, ...], [batch_size, 1, 1])
    distances_feature = tf.to_float(tf.one_hot(distances, 10))
    return distances_feature


def cluster_scores_layer(input_tensor,
                         end_scores,
                         speaker_ids,
                         genre,
                         attention_mask=None,
                         size_per_head=512,
                         initializer_range=0.02,
                         batch_size=None,
                         seq_length=None):
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    input_shape = get_shape_list(input_tensor, expected_rank=[2, 3])

    if len(input_shape) == 3:
        batch_size = input_shape[0]
        seq_length = input_shape[1]
    elif len(input_shape) == 2:
        if (batch_size is None or input_shape is None):
            raise ValueError(
                "When passing in rank 2 tensors to attention_layer, the values "
                "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                "must all be specified.")

    # [B, S, 31]
    mention_widths_feature = get_mention_width_feature(tf.nn.softmax(end_scores, axis=-1), batch_size, seq_length)

    # [B, S, S, 10]
    distances_feature = get_mention_distances_feature(batch_size, seq_length)

    same_speaker = tf.equal(speaker_ids[:, None, :], speaker_ids[:, :, None])
    same_speaker_feature = tf.one_hot(tf.to_int32(same_speaker), 2)

    genre_feature = tf.tile(tf.one_hot(genre, 7), [1, seq_length, 1])

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   S = sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    input_tensor_2d = reshape_to_matrix(input_tensor)

    # `features_layer` = [B*S, N*H]
    features_layer = tf.layers.dense(
        input_tensor_2d,
        size_per_head,
        activation=gelu,
        name="features",
        kernel_initializer=create_initializer(initializer_range))

    # `features_layer` = [B, 1, S, H]
    features_layer = transpose_for_scores(features_layer, batch_size, 1, seq_length, size_per_head)
    # `features_layer` = [B, S, H]
    features_layer = tf.squeeze(features_layer, 1)
    features_layer = tf.concat([features_layer, mention_widths_feature, genre_feature], axis=-1)

    features_a = tf.tile(tf.expand_dims(features_layer, 1), [1, seq_length, 1, 1])
    features_b = tf.tile(tf.expand_dims(features_layer, 2), [1, 1, seq_length, 1])

    cluster_features = tf.concat([features_a, features_b, distances_feature, same_speaker_feature], axis=-1)

    hidden_layer = tf.layers.dense(
        cluster_features,
        size_per_head,
        activation=gelu,
        name="hidden",
        kernel_initializer=create_initializer(initializer_range))

    # [B, S, S, 1]
    scores_layer = tf.layers.dense(
        hidden_layer,
        1,
        activation=None,
        name="scores",
        kernel_initializer=create_initializer(initializer_range))
    scores_layer = tf.squeeze(scores_layer, 3)

    if attention_mask is not None:
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        scores_layer = scores_layer * tf.cast(attention_mask, tf.float32) + adder

    return scores_layer


CLUSTER_THETA = 7


def get_coref_outputs(bert_config,
                      input_tensor,
                      input_mask,
                      speaker_ids,
                      genre,
                      mention_starts,
                      mention_ends_ids,
                      mention_ends_mask,
                      mention_clusters):
    """Get loss and outputs for coreference resolution."""
    with tf.variable_scope("coref"):
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length (= 1)
        #   T = `to_tensor` sequence length (= 10)
        #   N = `num_attention_heads` (= 1)
        #   H = `size_per_head` (= bert_config.hidden_size)

        print(FLAGS.cluster_loss_type)
        assert FLAGS.cluster_loss_type in ('backreference', 'hinge_loss')

        batch_size, seq_length, _ = modeling.get_shape_list(input_tensor, expected_rank=3)
        input_tensor_2d = reshape_to_matrix(input_tensor)
        input_mask_float = tf.to_float(input_mask)
        mention_ends_mask = mention_ends_mask * tf.expand_dims(input_mask, 1)

        with tf.variable_scope("mention_starts"):
            mention_starts_scores = tf.layers.dense(
                input_tensor,
                2,
                activation=None,
                name="output",
                kernel_initializer=modeling.create_initializer(0.02))

            mention_starts_loss = tf.losses.sparse_softmax_cross_entropy(
                mention_starts, mention_starts_scores, reduction='none')
            mention_starts_loss_pos = tf.boolean_mask(mention_starts_loss, tf.equal(mention_starts, 1) & tf.equal(input_mask, 1))
            mention_starts_loss_neg = tf.boolean_mask(mention_starts_loss, tf.equal(mention_starts, 0) & tf.equal(input_mask, 1))
            mention_starts_loss = 0.5 * (tf.reduce_mean(mention_starts_loss_pos) +
                                         tf.reduce_mean(mention_starts_loss_neg))
            # mention_starts_loss *= input_mask_float
            # mention_starts_loss = tf.reduce_sum(mention_starts_loss) / tf.reduce_sum(input_mask_float)

        with tf.variable_scope("mention_ends"):
            # [batch, 1, seq_length, seq_length]
            all_mention_ends_scores, ends_query, ends_key = attention_scores_layer(input_tensor_2d,
                                                                                   input_tensor_2d,
                                                                                   mention_ends_mask,
                                                                                   size_per_head=bert_config.hidden_size,
                                                                                   batch_size=batch_size,
                                                                                   from_seq_length=seq_length,
                                                                                   to_seq_length=seq_length,
                                                                                   return_features=True)
            ends_query = tf.squeeze(ends_query, 1)
            ends_key = tf.squeeze(ends_key, 1)

            # [batch, seq_length, seq_length]
            all_mention_ends_scores = tf.squeeze(all_mention_ends_scores, axis=[1])

            mention_ends_scores = tf.boolean_mask(all_mention_ends_scores, mention_starts)
            mention_ends_ids = tf.boolean_mask(mention_ends_ids, mention_starts)
            mention_ends_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mention_ends_ids,
                                                                               logits=mention_ends_scores,
                                                                               name='mention_ends_loss')
            mention_ends_loss = tf.reduce_sum(mention_ends_loss) / tf.to_float(tf.reduce_sum(mention_ends_mask))
            mention_ends_loss = mention_ends_loss * 100.

        with tf.variable_scope("clusters"):
            cluster_features = None
            # [batch_size, 1, seq_length, seq_length]
            if FLAGS.cluster_loss_type != 'backreference':
                cluster_mask_2d = input_mask[:, None, :] * input_mask[:, :, None]
            else:
                cluster_mask_2d = 1 - mention_ends_mask

            if FLAGS.add_cluster_features:
                mention_clusters_scores = cluster_scores_layer(input_tensor,
                                                               all_mention_ends_scores,
                                                               speaker_ids=speaker_ids,
                                                               genre=genre,
                                                               attention_mask=cluster_mask_2d,
                                                               size_per_head=bert_config.hidden_size)
            else:
                mention_clusters_scores, cluster_features, _ = attention_scores_layer(input_tensor_2d,
                                                                                      input_tensor_2d,
                                                                                      cluster_mask_2d,
                                                                                      size_per_head=bert_config.hidden_size,
                                                                                      batch_size=batch_size,
                                                                                      from_seq_length=seq_length,
                                                                                      to_seq_length=seq_length,
                                                                                      query_equals_key=True,
                                                                                      return_features=True)
                mention_clusters_scores = tf.squeeze(mention_clusters_scores, 1)
                cluster_features = tf.squeeze(cluster_features, 1)

            mention_clusters_float = tf.to_float(mention_clusters)

            # # softmax loss
            # per_example_cluster_size = tf.reduce_sum(mention_clusters_float, axis=[-1])
            # clusters_log_probs = tf.nn.log_softmax(mention_clusters_scores)  # [B,S,S]
            # per_example_clusters_loss = -tf.reduce_sum(
            #     tf.boolean_mask(clusters_log_probs, mention_starts) * tf.boolean_mask(mention_clusters_float,
            #                                                                           mention_starts),
            #     axis=[-1]) / tf.boolean_mask(per_example_cluster_size, mention_starts)

            # hinge loss
            if FLAGS.cluster_loss_type == 'hinge_loss':
                margin = 1
                t = tf.boolean_mask(mention_clusters_float, mention_starts) * 2 - 1
                y = tf.boolean_mask(mention_clusters_scores, mention_starts)
                # t = mention_clusters_float * 2 - 1
                # y = mention_clusters_scores
                per_example_clusters_loss = tf.maximum(0., margin - (t * (y - CLUSTER_THETA)))

                # # hard negative sampling
                # positive_losses = tf.boolean_mask(per_example_clusters_loss, t > 0)
                # negative_losses = tf.boolean_mask(per_example_clusters_loss, t < 0)
                # negative_losses, _ = tf.nn.top_k(negative_losses, k=tf.maximum(tf.shape(positive_losses)[0] * 5, 1))
                # per_example_clusters_loss = tf.concat([positive_losses, negative_losses], axis=0)

                mention_clusters_loss = tf.reduce_mean(per_example_clusters_loss)
            elif FLAGS.cluster_loss_type == 'backreference':
                # backward reference
                clusters_mask = 1 - mention_ends_mask
                mention_clusters = mention_clusters * clusters_mask
                is_first_item_in_cluster = tf.to_int32(tf.reduce_all(tf.equal(mention_clusters, 0), axis=-1))
                mention_clusters = tf.concat([tf.expand_dims(is_first_item_in_cluster, -1), mention_clusters], axis=-1)
                mention_clusters_float = tf.to_float(mention_clusters)
                epsilon_scores = tf.ones([batch_size, seq_length, 1], tf.float32) * CLUSTER_THETA
                mention_clusters_scores = tf.concat([epsilon_scores, mention_clusters_scores], axis=-1)

                clusters_probs = tf.nn.softmax(mention_clusters_scores, axis=-1)  # [B,S,S]
                correct_clusters_probs = clusters_probs * mention_clusters_float
                sum_correct_clusters_probs = tf.reduce_sum(correct_clusters_probs, axis=-1)  # [B,S]
                # sum_correct_clusters_probs = tf.boolean_mask(sum_correct_clusters_probs, mention_starts)
                mention_clusters_loss = tf.reduce_mean(-tf.log(sum_correct_clusters_probs + 1e-6))
                # mention_clusters_loss *= 0.1

    loss = mention_starts_loss + mention_ends_loss + mention_clusters_loss

    output = dict(loss=loss,
                  mention_starts_loss=mention_starts_loss,
                  mention_ends_loss=mention_ends_loss,
                  mention_clusters_loss=mention_clusters_loss,
                  mention_starts_scores=mention_starts_scores,
                  mention_ends_scores=all_mention_ends_scores,
                  mention_ends_query=ends_query,
                  mention_ends_key=ends_key,
                  mention_clusters_scores=mention_clusters_scores,
                  mention_clusters_features=cluster_features)

    return output


def get_coref_metrics(coref_outputs, input_mask, mention_starts, mention_ends_ids, mention_ends_mask, mention_clusters):
    mention_starts_scores = coref_outputs['mention_starts_scores']
    mention_ends_scores = coref_outputs['mention_ends_scores']
    mention_clusters_scores = coref_outputs['mention_clusters_scores']

    mention_starts_predictions = tf.argmax(mention_starts_scores, axis=-1)
    mention_starts_recall = tf.metrics.recall(labels=mention_starts,
                                              predictions=mention_starts_predictions,
                                              weights=input_mask)
    mention_starts_precision = tf.metrics.precision(labels=mention_starts,
                                                    predictions=mention_starts_predictions,
                                                    weights=input_mask)

    mention_ends_ids = tf.boolean_mask(mention_ends_ids, mention_starts, name='mention_ends_ids')
    mention_ends_scores = tf.boolean_mask(mention_ends_scores, mention_starts)
    mention_ends_predictions = tf.argmax(mention_ends_scores, axis=-1)
    mention_ends_accuracy = tf.metrics.accuracy(labels=mention_ends_ids,
                                                predictions=mention_ends_predictions)

    # # cluster accuracy - always 100% because it is easy to connect to yourself...
    if FLAGS.cluster_loss_type == 'backreference':
        batch_size, seq_length = modeling.get_shape_list(mention_starts, expected_rank=2)

        clusters_mask = 1 - mention_ends_mask
        mention_clusters = mention_clusters * clusters_mask
        is_first_item_in_cluster = tf.to_int32(tf.reduce_all(tf.equal(mention_clusters, 0), axis=-1))
        mention_clusters = tf.concat([tf.expand_dims(is_first_item_in_cluster, -1), mention_clusters], axis=-1)

        mention_clusters_predictions = tf.argmax(mention_clusters_scores, axis=-1)
        mention_clusters_predictions_one_hot = tf.one_hot(mention_clusters_predictions, depth=seq_length + 1)
        mention_clusters_per_mention_correct = tf.reduce_sum(
            mention_clusters_predictions_one_hot * tf.to_float(mention_clusters),
            axis=-1)
        mention_clusters_accuracy = tf.reduce_mean(tf.boolean_mask(mention_clusters_per_mention_correct, mention_starts))
        # mention_clusters_accuracy = tf.reduce_mean(mention_clusters_per_mention_correct)
        mention_clusters_recall = (mention_clusters_accuracy, mention_clusters_accuracy)
        mention_clusters_precision = (mention_clusters_accuracy, mention_clusters_accuracy)
    else:
        input_mask_2d = input_mask[:, None, :] * input_mask[:, :, None]
        mention_clusters_predictions = mention_clusters_scores > CLUSTER_THETA
        mention_clusters_recall = tf.metrics.recall(labels=mention_clusters,
                                                    predictions=mention_clusters_predictions,
                                                    weights=input_mask_2d)
        mention_clusters_precision = tf.metrics.precision(labels=mention_clusters,
                                                          predictions=mention_clusters_predictions,
                                                          weights=input_mask_2d)

    return dict(mention_starts_recall=mention_starts_recall,
                mention_starts_precision=mention_starts_precision,
                mention_ends_accuracy=mention_ends_accuracy,
                mention_clusters_recall=mention_clusters_recall,
                mention_clusters_precision=mention_clusters_precision)


def sort_cluster(clusters):
    clusters = sorted(clusters, key=lambda c: c[0][0])
    clusters = sorted(clusters, key=len, reverse=True)
    clusters = tuple(tuple(sorted(c, key=lambda x: x[0])) for c in clusters)
    return clusters


def cluster_matrix_to_clusters(mention_end_ids, cluster_matrix):
    mention_to_cluster = dict()
    for i, j in zip(*np.where(cluster_matrix)):
        mention1 = (i, mention_end_ids[i])
        mention2 = (j, mention_end_ids[j])
        if mention1 in mention_to_cluster and mention2 in mention_to_cluster:
            mention_to_cluster[mention1].update(mention_to_cluster[mention2])
            mention_to_cluster[mention2] = mention_to_cluster[mention1]
        elif mention1 in mention_to_cluster:
            mention_to_cluster[mention1].add(mention2)
            mention_to_cluster[mention2] = mention_to_cluster[mention1]
        elif mention2 in mention_to_cluster:
            mention_to_cluster[mention2].add(mention1)
            mention_to_cluster[mention1] = mention_to_cluster[mention2]
        else:
            new_cluster = {mention1, mention2}
            mention_to_cluster[mention1] = new_cluster
            mention_to_cluster[mention2] = new_cluster

    for m in mention_to_cluster:
        mention_to_cluster[m] = tuple(mention_to_cluster[m])

    clusters = set(map(tuple, mention_to_cluster.values()))

    return clusters, mention_to_cluster


def run_prediction_step(uf, mentions, document_offset, pred_start_scores, pred_end_scores, pred_cluster_scores, input_mask):
    pred_end_ids = pred_end_scores.argmax(-1)
    pred_mentions_mask = pred_start_scores.argmax(-1) * input_mask
    pred_cluster_scores[..., 1:] += (1. - pred_mentions_mask[None, :]) * -10000.

    for start in np.where(pred_mentions_mask)[0]:
        mention = (start + document_offset, pred_end_ids[start] + document_offset)
        mentions.add(mention)

        backref = pred_cluster_scores[start].argmax()

        # zero means we start a new cluster
        if backref > 0:
            backref_mention = (backref - 1 + document_offset, pred_end_ids[backref - 1] + document_offset)
            mentions.add(backref_mention)
            uf.unite(start + document_offset, backref + document_offset - 1)


def get_prediction_clusters(uf, mentions):
    predicted_clusters = []
    predicted_mention_to_cluster = {}

    end_at_start = np.zeros(10000, np.int32)
    for start, end in mentions:
        end_at_start[start] = end
    starts = list(set([m[0] for m in mentions]))

    for start in starts:
        if uf.parent[start] == start:
            cluster_starts = [i for i in range(max(starts) + 1) if uf.issame(start, i)]
            cluster_ends = [end_at_start[s] for s in cluster_starts]
            cluster = tuple(zip(cluster_starts, cluster_ends))
            predicted_clusters.append(cluster)
            for m in cluster:
                predicted_mention_to_cluster[m] = cluster

    return predicted_clusters, predicted_mention_to_cluster


class CorefPredictor:
    def __init__(self):
        self.cluster_features = None
        self.end_ids = None
        self.start_scores = None

    def reset(self):
        self.cluster_features = None
        self.end_ids = None
        self.start_scores = None

    def _update_overlapping_features(self, prev, cur, document_offset, mean=True):
        if prev is None:
            return cur
        else:
            prev_begin = prev[:document_offset]
            prev_overlap = prev[document_offset:]
            cur_overlap = cur[:len(prev_overlap)]
            cur_new = cur[len(prev_overlap):, ]
            if mean:
                updated_overlap = 0.5 * (prev_overlap + cur_overlap)
            else:
                updated_overlap = cur_overlap
            result = np.concatenate([prev_begin, updated_overlap, cur_new])
            return result

    def update(self, document_offset, start_scores, end_scores, cluster_features, input_mask):
        n_valid = input_mask.sum()
        start_scores = start_scores[:n_valid]
        end_scores = end_scores[:n_valid]
        cluster_features = cluster_features[:n_valid]

        self.cluster_features = self._update_overlapping_features(self.cluster_features, cluster_features, document_offset)
        end_ids = end_scores.argmax(-1) + document_offset
        self.end_ids = self._update_overlapping_features(self.end_ids, end_ids, document_offset, mean=False)
        self.start_scores = self._update_overlapping_features(self.start_scores, start_scores, document_offset)

    def get_clusters(self):
        assert len(self.start_scores) == len(self.end_ids) == len(self.cluster_features)

        document_len = len(self.cluster_features)
        hidden_size = self.cluster_features.shape[-1]

        ends_mask = create_mention_ends_mask(1, document_len)[0]
        clusters_mask = 1 - ends_mask
        mentions_mask = self.start_scores.argmax(-1)

        cluster_scores = np.dot(self.cluster_features, self.cluster_features.T) * clusters_mask
        cluster_scores /= math.sqrt(hidden_size)
        epsilon_scores = np.full([document_len, 1], CLUSTER_THETA)
        cluster_scores = np.concatenate([epsilon_scores, cluster_scores], axis=-1)

        mentions = set()
        uf = unionfind.unionfind(document_len)
        for start in np.where(mentions_mask)[0]:
            mention = (start, self.end_ids[start])
            mentions.add(mention)

            backref = cluster_scores[start].argmax()

            # zero means we start a new cluster
            if backref > 0:
                backref_mention = (backref - 1, self.end_ids[backref - 1])
                mentions.add(backref_mention)
                uf.unite(start, backref - 1)

        clusters = []
        mention_to_cluster = {}

        for start in np.where(mentions_mask)[0]:
            if uf.parent[start] == start:
                cluster_starts = [i for i in range(document_len) if uf.issame(start, i)]
                cluster_ends = [self.end_ids[s] for s in cluster_starts]
                cluster = tuple(zip(cluster_starts, cluster_ends))
                clusters.append(cluster)
                for m in cluster:
                    mention_to_cluster[m] = cluster

        return clusters, mention_to_cluster


def get_gold_clusters(example):
    gold_clusters = defaultdict(list)
    for i, s, e in zip(example.cluster_ids, example.gold_starts, example.gold_ends):
        gold_clusters[i].append((s, e))

    gold_clusters = tuple([tuple(c) for c in gold_clusters.values()])

    gold_mention_to_cluster = dict()
    for c in gold_clusters:
        for m in c:
            gold_mention_to_cluster[m] = c

    return gold_clusters, gold_mention_to_cluster


def evaluate_coref(pred_start_scores, pred_end_scores, pred_cluster_scores, input_mask, gold_end_ids, gold_cluster_matrix, evaluator):
    # create gold starts
    gold_clusters, gold_mention_to_cluster = cluster_matrix_to_clusters(gold_end_ids, gold_cluster_matrix)

    # # create pred starts
    if FLAGS.cluster_loss_type == 'backreference':
        pred_end_ids = pred_end_scores.argmax(-1)
        pred_mentions_mask = pred_start_scores.argmax(-1) * input_mask
        pred_cluster_scores[..., 1:] += (1. - pred_mentions_mask[None, :]) * -10000.
        uf = unionfind.unionfind(FLAGS.max_seq_length)
        for start in np.where(pred_mentions_mask)[0]:
            backref = pred_cluster_scores[start].argmax()

            # zero means we start a new cluster
            if backref > 0:
                uf.unite(start, backref - 1)

        predicted_clusters = []
        predicted_mention_to_cluster = {}
        for start in np.where(pred_mentions_mask)[0]:
            if uf.parent[start] == start:
                cluster_starts = [i for i in range(FLAGS.max_seq_length) if uf.issame(start, i)]
                cluster_ends = [pred_end_ids[s] for s in cluster_starts]
                cluster = tuple(zip(cluster_starts, cluster_ends))
                predicted_clusters.append(cluster)
                for m in cluster:
                    predicted_mention_to_cluster[m] = cluster

        # pred_end_ids = pred_end_scores.argmax(-1)
        # pred_mentions_mask = pred_start_scores.argmax(-1) * input_mask
        # pred_cluster_scores = pred_mentions_mask[:, None] * pred_mentions_mask[None, :] * pred_cluster_scores[..., 1:]
        # pred_cluster_matrix = np.zeros_like(pred_cluster_scores)
        # for i in np.where(pred_mentions_mask)[0]:
        #     if np.all(pred_cluster_scores[i] <= CLUSTER_THETA):
        #         pred_cluster_matrix[i, i] = 1
        #     else:
        #         ref = pred_cluster_scores[i].argmax()
        #         pred_cluster_matrix[i, ref] = 1
        # predicted_clusters, predicted_mention_to_cluster = cluster_matrix_to_clusters(pred_end_ids, pred_cluster_matrix)
    else:
        predicted_clusters = chinese_whispers(pred_start_scores, pred_end_scores, pred_cluster_scores, input_mask, CLUSTER_THETA)
        predicted_mention_to_cluster = {}
        for c in predicted_clusters:
            for m in c:
                predicted_mention_to_cluster[m] = c

    gold_clusters = sort_cluster(gold_clusters)
    predicted_clusters = sort_cluster(predicted_clusters)

    evaluator.update(predicted_clusters, gold_clusters, predicted_mention_to_cluster, gold_mention_to_cluster)
    return gold_clusters, predicted_clusters


FORES = [Fore.BLUE,
         Fore.CYAN,
         Fore.GREEN,
         Fore.MAGENTA,
         Fore.RED,
         Fore.YELLOW]
BACKS = [Back.BLUE,
         Back.CYAN,
         Back.GREEN,
         Back.MAGENTA,
         Back.RED,
         Back.YELLOW]
COLOR_WHEEL = FORES + [f + b for f in FORES for b in BACKS]


def coref_pprint(input_tokens, clusters):
    starts = set([m[0] for m in set(sum(clusters, ()))])
    start_to_cluster = {}
    cluster_to_color = {c: i % len(COLOR_WHEEL) for i, c in enumerate(clusters)}
    for c in clusters:
        for m in c:
            start_to_cluster[m[0]] = c
    ends = set([m[1] for m in set(sum(clusters, ()))])
    pretty_str = ''
    for i, t in enumerate(input_tokens):
        if t == '[PAD]':
            continue
        if i in starts:
            cluster = start_to_cluster[i]
            cluster_color = cluster_to_color[cluster]
            pretty_str += Style.BRIGHT + COLOR_WHEEL[cluster_color]

        pretty_str += t + ' '

        if i in ends:
            pretty_str += Style.RESET_ALL

    print(pretty_str)


def input_fn_builder(input_files,
                     max_seq_length,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "speaker_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "genre": tf.FixedLenFeature([1], tf.int64),
            "mention_starts": tf.FixedLenFeature([max_seq_length], tf.int64),
            "mention_ends_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "mention_clusters": tf.FixedLenFeature([max_seq_length, max_seq_length], tf.int64),
        }

        if FLAGS.do_pred:
            name_to_features["document_index"] = tf.FixedLenFeature([1], tf.int64)
            name_to_features["document_offset"] = tf.FixedLenFeature([1], tf.int64)

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
                tf.data.experimental.parallel_interleave(
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

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=False)

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
        use_one_hot_embeddings=FLAGS.use_tpu)

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
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
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

        tf.logging.set_verbosity(tf.logging.ERROR)

        with open('/specific/netapp5_2/gamir/benkantor/python/e2e-coref/test.english.jsonlines') as f:
            json_examples = [json.loads(jsonline) for jsonline in f.readlines()]
            examples = [process_example(je, i, should_filter_embedded_mentions=True).bertify(tokenizer) for i, je in enumerate(json_examples)]

        last_checkpoint = None
        while True:
            last_checkpoint = tf.contrib.training.wait_for_new_checkpoint(FLAGS.output_dir, last_checkpoint)

            # If TPU is not available, this will fall back to normal Estimator on CPU
            # or GPU.
            estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=FLAGS.use_tpu,
                model_fn=model_fn,
                config=run_config,
                train_batch_size=FLAGS.train_batch_size,
                eval_batch_size=FLAGS.eval_batch_size,
                predict_batch_size=FLAGS.eval_batch_size)

            eval_input_fn = input_fn_builder(
                input_files=input_files,
                max_seq_length=FLAGS.max_seq_length,
                is_training=False)

            predictions = estimator.predict(input_fn=eval_input_fn)

            global_step = estimator.get_variable_value("global_step")
            print('*' * 128)
            print(' GLOBAL STEP {} '.format(global_step).center(128, '*'))
            print('*' * 128)

            evaluator = metrics.CorefEvaluator()
            stats = OrderedDict()
            prev_index = -1
            uf = unionfind.unionfind(10000)
            mentions = set()
            # coref_predictor = CorefPredictor()
            for p in predictions:
                stats['mention_starts_recall'] = float(p['mention_starts_recall'])
                stats['mention_starts_precision'] = float(p['mention_starts_precision'])
                stats['mention_ends_accuracy'] = float(p['mention_ends_accuracy'])
                stats['mention_clusters_recall'] = float(p['mention_clusters_recall'])
                stats['mention_clusters_precision'] = float(p['mention_clusters_precision'])

                document_index = p['document_index'][0]
                document_offset = p['document_offset'][0]
                pred_start_scores = p['pred_start_scores']
                pred_end_scores = p['pred_end_scores']
                pred_end_query = p['pred_end_query']
                pred_end_key = p['pred_end_key']
                pred_cluster_scores = p['pred_cluster_scores']
                pred_cluster_features = p['pred_cluster_features']
                input_mask = p['input_mask']
                gold_end_ids = p['gold_end_ids']
                gold_clusters_matrix = p['gold_clusters']

                if document_index == prev_index:
                    # coref_predictor.update(document_offset, pred_start_scores, pred_end_scores,
                    #                        pred_cluster_features, input_mask)
                    run_prediction_step(uf, mentions, document_offset, pred_start_scores, pred_end_scores,
                                        pred_cluster_scores, input_mask)
                else:
                    assert document_index == prev_index + 1
                    if prev_index >= 0:
                        gold_clusters, gold_mention_to_cluster = get_gold_clusters(examples[prev_index])
                        # predicted_clusters, predicted_mention_to_cluster = coref_predictor.get_clusters()
                        predicted_clusters, predicted_mention_to_cluster = get_prediction_clusters(uf, mentions)
                        # gold_clusters, predicted_clusters = evaluate_coref(pred_start_scores,
                        #                                                    pred_end_scores,
                        #                                                    pred_cluster_scores,
                        #                                                    input_mask,
                        #                                                    gold_end_ids,
                        #                                                    gold_clusters_matrix,
                        #                                                    evaluator)

                        gold_clusters = sort_cluster(gold_clusters)
                        predicted_clusters = sort_cluster(predicted_clusters)

                        evaluator.update(predicted_clusters, gold_clusters, predicted_mention_to_cluster,
                                         gold_mention_to_cluster)

                        if FLAGS.print_results:
                            print('GOLD CLUSTERING:', end='\t')
                            coref_pprint(examples[prev_index].tokens, gold_clusters)
                            print('PREDICTED CLUSTERING:', end='\t')
                            coref_pprint(examples[prev_index].tokens, predicted_clusters)
                            print('==================================================================')

                    prev_index = document_index
                    # coref_predictor.reset()
                    # coref_predictor.update(document_offset, pred_start_scores, pred_end_scores,
                    #                        pred_cluster_features, input_mask)
                    uf = unionfind.unionfind(10000)
                    mentions = set()
                    run_prediction_step(uf, mentions, document_offset, pred_start_scores, pred_end_scores,
                                        pred_cluster_scores, input_mask)

            # TODO: insert to function?
            gold_clusters, gold_mention_to_cluster = get_gold_clusters(examples[document_index])
            # predicted_clusters, predicted_mention_to_cluster = coref_predictor.get_clusters()
            predicted_clusters, predicted_mention_to_cluster = get_prediction_clusters(uf, mentions)
            gold_clusters = sort_cluster(gold_clusters)
            predicted_clusters = sort_cluster(predicted_clusters)
            evaluator.update(predicted_clusters, gold_clusters, predicted_mention_to_cluster,
                             gold_mention_to_cluster)

            p, r, f = evaluator.get_prf()
            stats['Average F1 (py)'] = f
            stats['Average precision (py)'] = p
            stats['Average recall (py)'] = r

            for k, v in stats.items():
                print('{}: {:.2f}%'.format(k, v * 100))

            _write_dict_to_summary(FLAGS.output_dir, stats, global_step)

            evaluator.pprint()

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
