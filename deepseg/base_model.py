# Copyright 2018 luozhouyang.
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
# ==============================================================================

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from . import collections
from . import dataset_util
from .abstract_model import AbstractModel


class BaseModel(AbstractModel):
    """Base model. Word embedding + BiLSTM."""

    def input_fn(self, params, mode=tf.estimator.ModeKeys.TRAIN):
        return dataset_util.build_dataset(params, mode)

    def model_fn(self, features, labels, mode, params, config):
        # [batch_size, world_length]
        words = features['inputs']
        nwords = features['inputs_length']
        # a UNK token should placed in the first row in vocab file
        words_str2idx = lookup_ops.index_table_from_file(
            params['src_vocab'], default_value=0)
        # [batch_size, world_length]
        words_ids = words_str2idx.lookup(words)

        training = mode == tf.estimator.ModeKeys.TRAIN

        # embedding
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            # [vocab_size, embedding_size]
            variable = tf.get_variable(
                "words_embedding",
                shape=(params['vocab_size'], params['embedding_size']),
                dtype=tf.float32)
            # [batch_size, world_length, embedding_size]
            embedding = tf.nn.embedding_lookup(variable, words_ids)
            embedding = tf.layers.dropout(
                embedding, rate=params['dropout'], training=training)

        # BiLSTM
        with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
            # transpose embedding for time major mode
            # [world_length, batch_size, embedding_size]
            inputs = tf.transpose(embedding, perm=[1, 0, 2])
            lstm_fw = tf.nn.rnn_cell.LSTMCell(params['lstm_size'])
            lstm_bw = tf.nn.rnn_cell.LSTMCell(params['lstm_size'])

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw,
                cell_bw=lstm_bw,
                inputs=inputs,
                sequence_length=nwords,
                dtype=tf.float32,
                swap_memory=True,
                time_major=True)

            # [len, 32, 256*2=512]
            output = tf.concat([output_fw, output_bw], axis=-1)
            # [32, len, 512]
            output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.layers.dropout(
                output, rate=params['dropout'], training=training)

        # logits : [32, len, 5]
        # predict_ids : [32, len]
        logits, predict_ids = self.decode(output, nwords, params)

        # TODO(luozhouyang) Add hooks
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.build_predictions(predict_ids, params)
            prediction_hooks = []
            key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            export_outputs = {
                key: tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs,
                prediction_hooks=prediction_hooks)

        # loss = self.compute_loss(logits, labels, nwords, params)

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = self.build_eval_metrics(
                predict_ids, labels, nwords, params)
            predictions = self.build_predictions(predict_ids, params)
            if predictions:
                collections.add_dict_to_collection("predictions", predictions)
            eval_hooks = []
            loss = self.compute_loss(logits, labels, nwords, params)
            tf.add_to_collection(collections.EVAL_LOSS, loss)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=metrics,
                evaluation_hooks=eval_hooks)

        if mode == tf.estimator.ModeKeys.TRAIN:
            loss = self.compute_loss(logits, labels, nwords, params)
            tf.add_to_collection(collections.TRAIN_LOSS, loss)
            train_op = self.build_train_op(loss, params)
            train_hooks = []
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks)

    def decode(self, output, nwords, params):
        raise NotImplementedError()

    def compute_loss(self, logits, labels, nwords, params):
        raise NotImplementedError()

    def build_predictions(self, predict_ids, params):
        tags_idx2str = lookup_ops.index_to_string_table_from_file(
            params['tag_vocab'], default_value=params['oov_tag'])
        predict_tags = tags_idx2str.lookup(tf.cast(predict_ids, tf.int64))
        predictions = {
            "predict_ids": predict_ids,
            "predict_tags": predict_tags
        }
        return predictions

    def build_eval_metrics(self, predict_ids, labels, nwords, params):
        tags_str2idx = lookup_ops.index_table_from_file(
            params['tag_vocab'], default_value=0)
        actual_ids = tags_str2idx.lookup(labels)
        weights = tf.sequence_mask(nwords)
        metrics = {
            "accuracy": tf.metrics.accuracy(actual_ids, predict_ids, weights)
        }
        return metrics

    def build_train_op(self, loss, params):
        global_step = tf.train.get_or_create_global_step()
        if params['optimizer'].lower() == 'adam':
            opt = tf.train.AdamOptimizer()
            return opt.minimize(loss, global_step=global_step)
        if params['optimizer'].lower() == 'momentum':
            opt = tf.train.MomentumOptimizer(
                learning_rate=params.get('learning_rate', 1.0),
                momentum=params['momentum'])
            return opt.minimize(loss, global_step=global_step)
        if params['optimizer'].lower() == 'adadelta':
            opt = tf.train.AdadeltaOptimizer()
            return opt.minimize(loss, global_step=global_step)
        if params['optimizer'].lower() == 'adagrad':
            opt = tf.train.AdagradOptimizer(
                learning_rate=params.get('learning_rate', 1.0))
            return opt.minimize(loss, global_step=global_step)

        # TODO(luozhouyang) decay lr
        sgd = tf.train.GradientDescentOptimizer(
            learning_rate=params.get('learning_rate', 1.0))
        return sgd.minimize(loss, global_step=global_step)
