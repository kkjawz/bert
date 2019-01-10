import tensorflow as tf

print('n_train:', sum(1 for _ in tf.python_io.tf_record_iterator('/specific/netapp5_2/gamir/benkantor/python/bert/coref/train.english.tfrecords')))
print('n_dev:', sum(1 for _ in tf.python_io.tf_record_iterator('/specific/netapp5_2/gamir/benkantor/python/bert/coref/dev.english.tfrecords')))
print('n_test:', sum(1 for _ in tf.python_io.tf_record_iterator('/specific/netapp5_2/gamir/benkantor/python/bert/coref/test.english.tfrecords')))