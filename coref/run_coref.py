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
import tensorflow as tf
import numpy as np

import modeling
import optimization
import tokenization
from modeling import reshape_to_matrix, attention_scores_layer

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


def create_mention_ends_mask(batch_size):
    mention_ends_mask = np.zeros([FLAGS.max_seq_length, FLAGS.max_seq_length], np.int32)
    for i in range(FLAGS.max_seq_length):
        mention_ends_mask[i, i:] = 1
    mention_ends_mask = np.tile(mention_ends_mask[None, :, :], [batch_size, 1, 1])

    return tf.constant(mention_ends_mask, name='mention_ends_mask')


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
        mention_starts = features["mention_starts"]
        mention_ends_ids = features["mention_ends_ids"]
        mention_clusters = features["mention_clusters"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        mention_ends_mask = create_mention_ends_mask(FLAGS.train_batch_size if is_training else FLAGS.eval_batch_size)

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
            mention_starts=mention_starts,
            mention_ends_ids=mention_ends_ids,
            mention_ends_mask=mention_ends_mask,
            mention_clusters=mention_clusters)

        coref_metrics = get_coref_metrics(
            coref_outputs=coref_outputs,
            input_mask=input_mask,
            mention_starts=mention_starts,
            mention_ends_ids=mention_ends_ids,
            mention_clusters=mention_clusters)

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
                                                       'ca': coref_metrics['mention_clusters_accuracy']},
                                                      every_n_iter=10)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[logging_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:

            eval_metrics_ops = {'mention_starts_recall': coref_metrics['mention_starts_recall'],
                                'mention_starts_precision': coref_metrics['mention_starts_precision'],
                                'mention_ends_accuracy': coref_metrics['mention_ends_accuracy'],
                                'mention_clusters_accuracy': (coref_metrics['mention_clusters_accuracy'], tf.no_op())}
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=(lambda: eval_metrics_ops, []),
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            raise NotImplementedError()
        else:
            raise ValueError("Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_coref_outputs(bert_config, input_tensor, input_mask, mention_starts, mention_ends_ids, mention_ends_mask,
                      mention_clusters):
    """Get loss and outputs for coreference resolution."""
    with tf.variable_scope("coref"):
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   F = `from_tensor` sequence length (= 1)
        #   T = `to_tensor` sequence length (= 10)
        #   N = `num_attention_heads` (= 1)
        #   H = `size_per_head` (= bert_config.hidden_size)

        batch_size, seq_length, _ = modeling.get_shape_list(input_tensor, expected_rank=3)
        input_tensor_2d = reshape_to_matrix(input_tensor)
        input_mask_float = tf.to_float(input_mask)

        with tf.variable_scope("mention_starts"):
            mention_starts_scores = tf.layers.dense(
                input_tensor,
                2,
                activation=None,
                name="output",
                kernel_initializer=modeling.create_initializer(0.02))

            mention_starts_loss = tf.losses.sparse_softmax_cross_entropy(
                mention_starts, mention_starts_scores, reduction='none')
            mention_starts_loss *= input_mask_float
            mention_starts_loss = tf.reduce_sum(mention_starts_loss) / tf.reduce_sum(input_mask_float)

        with tf.variable_scope("mention_ends"):
            mention_ends_mask = mention_ends_mask * tf.expand_dims(input_mask, 1)
            # [batch, 1, seq_length, seq_length]
            all_mention_ends_scores = attention_scores_layer(input_tensor_2d,
                                                             input_tensor_2d,
                                                             mention_ends_mask,
                                                             size_per_head=bert_config.hidden_size,
                                                             batch_size=batch_size,
                                                             from_seq_length=seq_length,
                                                             to_seq_length=seq_length)
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
            mention_clusters_float = tf.to_float(mention_clusters)
            # [batch_size, 1, seq_length, seq_length]
            mention_clusters_scores = attention_scores_layer(input_tensor_2d,
                                                             input_tensor_2d,
                                                             # tf.expand_dims(input_mask, 1),
                                                             size_per_head=bert_config.hidden_size,
                                                             batch_size=batch_size,
                                                             from_seq_length=seq_length,
                                                             to_seq_length=seq_length)
            mention_clusters_scores = tf.squeeze(mention_clusters_scores, 1)

            # per_example_cluster_size = tf.reduce_sum(mention_clusters_float, axis=[-1], keep_dims=True)
            # mention_clusters_labels = mention_clusters_float / per_example_cluster_size
            # mention_clusters_loss = tf.nn.softmax_cross_entropy_with_logits(labels=mention_clusters_labels,
            #                                                                 logits=mention_clusters_scores,
            #                                                                 name='mention_clusters_loss')
            # mention_clusters_loss = tf.reduce_sum(mention_clusters_loss) / tf.reduce_sum(input_mask_float)

            per_example_cluster_size = tf.reduce_sum(mention_clusters_float, axis=[-1])
            clusters_log_probs = tf.nn.log_softmax(mention_clusters_scores)  # [B,S,S]

            per_example_clusters_loss = -tf.reduce_sum(
                tf.boolean_mask(clusters_log_probs, mention_starts) * tf.boolean_mask(mention_clusters_float,
                                                                                      mention_starts),
                axis=[-1]) / tf.boolean_mask(per_example_cluster_size, mention_starts)

            mention_clusters_loss = tf.reduce_mean(per_example_clusters_loss)

    loss = mention_starts_loss + mention_ends_loss + mention_clusters_loss

    output = dict(loss=loss,
                  mention_starts_loss=mention_starts_loss,
                  mention_ends_loss=mention_ends_loss,
                  mention_clusters_loss=mention_clusters_loss,
                  mention_starts_scores=mention_starts_scores,
                  mention_ends_scores=mention_ends_scores,
                  mention_clusters_scores=mention_clusters_scores)

    return output


def get_coref_metrics(coref_outputs, input_mask, mention_starts, mention_ends_ids, mention_clusters):
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
    mention_ends_predictions = tf.argmax(mention_ends_scores, axis=-1)
    mention_ends_accuracy = tf.metrics.accuracy(labels=mention_ends_ids,
                                                predictions=mention_ends_predictions)

    mention_clusters_predictions = tf.argmax(mention_clusters_scores, axis=-1)
    mention_clusters_predictions_one_hot = tf.one_hot(mention_clusters_predictions, depth=FLAGS.max_seq_length)
    mention_clusters_per_mention_correct = tf.reduce_sum(
        mention_clusters_predictions_one_hot * tf.to_float(mention_clusters),
        axis=-1)
    mention_clusters_accuracy = tf.reduce_sum(mention_clusters_per_mention_correct) / tf.to_float(
        tf.reduce_sum(mention_starts))

    return dict(mention_starts_recall=mention_starts_recall,
                mention_starts_precision=mention_starts_precision,
                mention_ends_accuracy=mention_ends_accuracy,
                mention_clusters_accuracy=mention_clusters_accuracy)


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
            "mention_starts": tf.FixedLenFeature([max_seq_length], tf.int64),
            "mention_ends_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "mention_clusters": tf.FixedLenFeature([max_seq_length, max_seq_length], tf.int64),
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

        tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=True)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
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

                probabilities = [math.exp(l) for l in log_probabilities]
                candidate_print = ', '.join(f'{c}: {p:.2f}' for c, p in zip(candidates, probabilities))
                print('candidates:', candidate_print)
                print('')


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
