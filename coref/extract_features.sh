export BERT_MODEL_PATH=/specific/netapp5_2/gamir/benkantor/models/bert/cased_L-24_H-1024_A-16
PYTHONPATH=. python coref/extract_features.py --input_file=/specific/netapp5_2/gamir/benkantor/python/e2e-coref/test.english.jsonlines --output_file=/specific/netapp5_2/gamir/benkantor/python/bert/coref/data/test.english.bert_features.hdf5 --bert_config_file $BERT_MODEL_PATH/bert_config.json --init_checkpoint $BERT_MODEL_PATH/bert_model.ckpt --vocab_file  $BERT_MODEL_PATH/vocab.txt --do_lower_case=False
