import tensorflow as tf

print('n_train:', sum(1 for _ in tf.python_io.tf_record_iterator('/specific/netapp5_2/gamir/benkantor/python/bert/train_no_overlap.english.tfrecords')))
print('n_dev:', sum(1 for _ in tf.python_io.tf_record_iterator('/specific/netapp5_2/gamir/benkantor/python/bert/dev_no_overlap.english.tfrecords')))
print('n_test:', sum(1 for _ in tf.python_io.tf_record_iterator('/specific/netapp5_2/gamir/benkantor/python/bert/test_no_overlap.english.tfrecords')))