import os
import pickle

import tensorflow as tf

tf.enable_eager_execution()

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

with open(os.path.expandvars('$DATA/CBTest/data/out_of_vocab.pkl'), 'rb') as f:
    out_of_vocab = pickle.load(f)

vocab_all = []
vocab_train = []

with open(os.path.expandvars('$DATA/CBTest/data/cbtest_NE_vocab_all.txt')) as f:
    for l in f:
        w = l.strip()
        if w:
            vocab_all.append(w)

with open(os.path.expandvars('$DATA/CBTest/data/cbtest_NE_vocab.txt')) as f:
    for l in f:
        w = l.strip()
        if w:
            vocab_train.append(w)

vocab_train = set(vocab_train)

out_of_vocab = []
for i, w in enumerate(vocab_all):
    if w not in vocab_train:
        out_of_vocab.append(i)

max_seq_length = 512
max_predictions_per_seq = 1
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
}

test_dataset = tf.data.TFRecordDataset([os.path.expandvars('/home/benk/data/CBTest/data/cbtest_NE_valid_2000ex_pos.tfrecords')])
test_dataset = tf.data.TFRecordDataset([os.path.expandvars('/home/benk/data/CBTest/data/cbtest_NE_test_2500ex_pos.tfrecords')])
# d = tf.data.TFRecordDataset([os.path.expandvars('/home/benk/data/CBTest/data/cbtest_NE_train.tfrecords')])
# Since we evaluate for a fixed number of steps we don't want to encounter
# out-of-range exceptions.
# d = d.repeat()

out_of_vocab = tf.constant(out_of_vocab)

d = d.map(lambda record: _decode_record(record, name_to_features))
d = d.filter(lambda e: tf.logical_not(tf.reduce_any(tf.equal(out_of_vocab, e["masked_lm_ids"][0]))))
# d = d.filter(lambda e: tf.logical_not(tf.reduce_any(tf.equal(out_of_vocab, e["masked_lm_ids"][0]))))

i = 0
for e in d:
    i += 1

print(i)
