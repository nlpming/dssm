# coding:utf-8
#############################################
# FileName: convert_data.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: convert tfrecord
#############################################
import json
import codecs
import numpy as np
import tensorflow as tf
from tqdm import tqdm

"""
转换为tfrecord格式
"""
def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def sent2id(sent, vocab, max_size):
    sent = [vocab[c] for c in sent if c in vocab]
    sent = sent[:max_size] + [0]*(max_size - len(sent))
    return sent

def convert_tfrecord(in_file, out_file, vocab_path, query_size=50, doc_size=200):
    vocab = json.load(codecs.open(vocab_path, "r", "utf-8"))
    writer = tf.io.TFRecordWriter(out_file)
    icount = 0
    with codecs.open(in_file, "r", "utf-8") as fr:
        for line in tqdm(fr):
            icount += 1
            data = json.loads(line)
            # 输入数据：query,doc_pos,doc_neg
            query = sent2id(data["query"], vocab, query_size)
            doc_pos = sent2id(data["doc_pos"], vocab, doc_size)
            doc_neg = sent2id(data["doc_neg"], vocab, doc_size)

            feed_dict = {
                "query": create_int_feature(query),
                "doc_pos": create_int_feature(doc_pos),
                "doc_neg": create_int_feature(doc_neg),
                "label": create_int_feature([1]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feed_dict))
            serialized = example.SerializeToString()
            writer.write(serialized)
    print(icount)
    writer.close()



if __name__ == "__main__":
    convert_tfrecord("../data/data_triplet.json",
                     "../data/train_triplet.tfrecord",
                     "../char.json",
                     query_size=50, 
                     doc_size=200)
