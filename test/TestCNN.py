import tensorflow as tf


class TestCNN(tf.test.TestCase):

    def testModel(self):
        # 构建数据通道
        dataset_features = tf.data.Dataset.from_tensor_slices(tf.random_uniform(shape=[1000, 28, 28, 1],
                                                                                maxval=255,
                                                                                dtype=tf.double))
        dataset_labels = tf.data.Dataset.from_tensor_slices(tf.random_uniform(shape=[1000, 1],
                                                                              maxval=9,
                                                                              dtype=tf.double))
        dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))
        dataset = dataset.map(
            lambda f, l: (tf.log(f), tf.cast(tf.log(l), dtype=tf.int32))
        )
        dataset = dataset.batch(2)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
        features, labels = iterator.get_next()

        input_layer = tf.reshape(features, shape=[-1, 28, 28, 1])
        # [batch_size, 28, 28, 32]
        convolution_first = tf.layers.conv2d(inputs=input_layer,
                                             filters=32,
                                             kernel_size=[5, 5],
                                             padding="same",
                                             activation=tf.nn.relu)
        # [batch_size, 14, 14, 32]
        pooling_first = tf.layers.max_pooling2d(inputs=convolution_first,
                                                pool_size=[2, 2],
                                                strides=2)

        # [batch_size, 14, 14, 64]
        convolution_second = tf.layers.conv2d(inputs=pooling_first,
                                              filters=64,
                                              kernel_size=[5, 5],
                                              padding="same",
                                              activation=tf.nn.relu)
        # [batch_size, 7, 7, 64]
        pooling_second = tf.layers.max_pooling2d(inputs=convolution_second,
                                                 pool_size=[2, 2],
                                                 strides=2)
        # [batch_size, 7*7*64]
        pooling_flat = tf.reshape(pooling_second,
                                  shape=[-1, 7*7*64])
        # [batch_size, 1024]
        dense = tf.layers.dense(inputs=pooling_flat,
                                units=1024,
                                activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense,
                                    rate=0.5,
                                    training=True)
        # [batch_size, 10]
        logits = tf.layers.dense(inputs=dropout, units=10)
        classes = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits=logits)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS))
            # print(sess.run(features))
            # print(sess.run(labels))
            # print(sess.run(tf.shape(input_layer)))
            # print(sess.run(tf.shape(convolution_first)))
            # print(sess.run(tf.shape(pooling_first)))
            # print(sess.run(tf.shape(convolution_second)))
            # print(sess.run(tf.shape(pooling_second)))
            # print(sess.run(tf.shape(pooling_flat)))
            # print(sess.run(tf.shape(dense)))
            # print(sess.run(tf.shape(logits)))
            # print(sess.run(logits))
            # print(sess.run(classes))
            # print(sess.run(probabilities))
            # print(sess.run(loss))

            # print(sess.run(dataset1))


if __name__ == '__main__':
    tf.test.main()
