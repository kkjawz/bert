import json
import tensorflow as tf

import tokenization
import numpy as np
from coref.data import process_example, flatten, filter_embedded_mentions
from visdom import Visdom

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")
flags.DEFINE_string("vocab_file", None, "")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=False)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    token_lengths = []
    mention_distances = []
    n_mentions = []
    n_filtered_mentions = 0
    total_mentions = 0
    for input_file in input_files:
        with open(input_file) as f:
            json_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        for json_e in json_examples:
            clusters = json_e["clusters"]
            gold_mentions = sorted(tuple(m) for m in flatten(clusters))
            filtered_gold_mentions = filter_embedded_mentions(gold_mentions)

            e = process_example(json_e)
            e = e.bertify(tokenizer)
            n_mentions.append(len(e.gold_starts))
            n_filtered_mentions += len(filtered_gold_mentions)
            total_mentions += len(gold_mentions)
            token_lengths.append(len(e.tokens))
            clusters = {i: np.where(e.cluster_ids == i)[0] for i in set(e.cluster_ids.astype(np.int32))}
            for i in clusters:
                cluster_starts = e.gold_starts[clusters[i]]
                cluster_starts = np.sort(cluster_starts)
                dists = cluster_starts[1:] - cluster_starts[:-1]
                mention_distances.extend(dists)

    viz = Visdom(port=8097, server="http://localhost")
    assert viz.check_connection(), 'No connection could be formed quickly'

    viz.histogram(X=n_mentions, opts=dict(numbins=200, title='n_mentions'))
    viz.histogram(X=token_lengths, opts=dict(numbins=200, title='token_lengths'))
    viz.histogram(X=mention_distances, opts=dict(numbins=200, title='mention_distances'))

    mention_distances = np.array(mention_distances)
    print('mention_distances > 256:', (mention_distances > 200).sum() / len(mention_distances))
    print('mention_distances > 512:', (mention_distances > 500).sum() / len(mention_distances))
    print('mention_distances > 1024:', (mention_distances > 500).sum() / len(mention_distances))
    print('%filtered_mentions:', n_filtered_mentions / total_mentions)


if __name__ == "__main__":
    tf.app.run()
