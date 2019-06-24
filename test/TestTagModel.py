import tensorflow as tf
from tensorflow.contrib import lookup as lookup_op

import os

DATA_DIR = "/Users/goofy/Desktop/DL/tag"
VOCAB_FEATURE = os.path.join(DATA_DIR, "vocab.feature.txt")
VOCAB_LABEL = os.path.join(DATA_DIR, "vocab.label.txt")
FEATURE_FILE = os.path.join(DATA_DIR, "feature.txt")
LABEL_FILE = os.path.join(DATA_DIR, "label.txt")


def transformer(x):
    tf.cast(x, dtype=tf.double)
    # with tf.Session() as s:
    #     print(s.run(tf.shape(x)))
    return x[:, 0:-1], x[:, -1]


class Test(tf.test.TestCase):

    def testDataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(tf.random_normal(shape=[32, 6, 5],
                                                                      mean=0,
                                                                      stddev=1.0,
                                                                      dtype=tf.float32))
        dataset = dataset.map(transformer)
        dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(2)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        data = iterator.get_next()
        feature, label = data

        with self.test_session() as sess:
            sess.run(iterator.initializer)
            # sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
            print(sess.run(feature))
            print(sess.run(tf.shape(feature)))
            print(sess.run(label))
            print(sess.run(tf.shape(label)))

    def testModel(self):
        # 数据输入
        dataset_f = tf.data.TextLineDataset(filenames=FEATURE_FILE)
        dataset_l = tf.data.TextLineDataset(filenames=LABEL_FILE)
        dataset = tf.data.Dataset.zip((dataset_f, dataset_l))
        dataset = dataset.map(
            lambda x, y: (tf.string_split([x], delimiter=",").values, tf.string_split([y], delimiter=",").values)
        )
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(
            lambda x, y: (x, tf.size(x), y)
        )
        dataset = dataset.padded_batch(batch_size=2, padded_shapes=([None, ], [], [None]), padding_values=(
            tf.constant("<UNK>", dtype=tf.string), 0, tf.constant("O", dtype=tf.string)))
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        feature, feature_length, label = iterator.get_next()

        # 通过字典获得索引
        words2idx = lookup_op.index_table_from_file(VOCAB_FEATURE)
        words_ids = words2idx.lookup(feature)

        # 编码
        with tf.variable_scope("embedding_test", reuse=tf.AUTO_REUSE):
            embedding_variable = tf.get_variable("embedding_variable", shape=[18, 256], dtype=tf.double)
            embedding = tf.nn.embedding_lookup(embedding_variable, words_ids)
            embedding_outputs = tf.layers.dropout(inputs=embedding, rate=0.5, training=True)
            inputs = embedding_outputs
        # 模型
        with tf.variable_scope("lstm_model", reuse=tf.AUTO_REUSE):
            cells = [tf.nn.rnn_cell.LSTMCell(256), tf.nn.rnn_cell.LSTMCell(256)]
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells)
            outputs, state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=inputs,
                sequence_length=feature_length,
                dtype=inputs.dtype
            )

        # 输出层
        logits = tf.layers.dense(inputs=outputs, units=5)
        # 解码
        transition = tf.get_variable(name="transition", shape=[5, 5], dtype=tf.double)
        decode_ids, best_score = tf.contrib.crf.crf_decode(logits, transition, feature_length)
        # 通过索引获得标签
        idx2tags = lookup_op.index_to_string_table_from_file(VOCAB_LABEL, default_value="O")
        tags = idx2tags.lookup(tf.cast(decode_ids, dtype=tf.int64))

        # 计算损失
        tags2idx = lookup_op.index_table_from_file(VOCAB_LABEL)
        actual_ids = tags2idx.lookup(label)
        log_likeilhood, _ = tf.contrib.crf.crf_log_likelihood(inputs=tf.cast(logits, dtype=tf.float32), tag_indices=actual_ids, sequence_lengths=feature_length)
        loss = tf.reduce_mean(-log_likeilhood)

        with self.test_session() as sess:
            sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
            sess.run(tf.global_variables_initializer())
            # print(sess.run(feature))
            # print(sess.run(feature_length))
            # print(sess.run(label))
            # print(sess.run(words2idx))
            # print(sess.run(words_ids))
            # print(sess.run(embedding_variable))
            # print(sess.run(tf.shape(embedding_variable)))
            # print(sess.run(embedding))
            # print(sess.run(tf.shape(embedding)))
            # print(sess.run(embedding_outputs))
            # print(sess.run(tf.shape(embedding_outputs)))
            # print(sess.run(outputs))
            # print(sess.run(tf.shape(outputs)))
            # print(sess.run(tf.shape(state[0])))
            # print(sess.run(logits))
            # print(sess.run(tf.shape(logits)))
            # print(sess.run(decode_tags))
            # print(sess.run(tags))
            # print(sess.run(actual_ids))
            # print(sess.run(log_likeilhood))
            print(sess.run(loss))

        return
