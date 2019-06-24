import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
import collections
import re
import os


class MyModel(object):
    def __init__(self, model_name="deepseg_test", host="10.100.3.200", port=8093):
        self.model_name = model_name
        self.host = host
        self.port = port

    def serving(self):
        raise NotImplementedError()


class ReloadModel(MyModel):
    def __init__(self, model_name="deepseg_test", host="10.100.3.200", port=8093, input_file=None, output_file=None):
        super(ReloadModel, self).__init__(model_name, host, port)
        self.channel = implementations.insecure_channel(host, port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(self.channel)
        self.input_file = input_file
        self.output_file = output_file
        self.pattern = re.compile("[0-9]+")

    @staticmethod
    def _translate_tag_to_string(src_list, tag_list):
        if not src_list or not tag_list:
            return False, None
        if len(src_list) != len(tag_list):
            return False, None
        result = src_list.copy()
        for _i in range(len(tag_list)):
            if tag_list[_i].lower() == "e" or tag_list[_i].lower() == "s":
                result[_i] += " "
            else:
                pass
        return "".join(result).strip()

    @staticmethod
    def _write_vocab(vocabs, file):
        vocabs = sorted(vocabs)
        with open(file, mode="wt", encoding="utf8", buffering=8192) as f:
            for v in vocabs:
                f.write(v + "\n")

    def serving(self):
        lines = 0
        counter = collections.Counter()
        with open(self.input_file, 'rt', encoding='utf8', buffering=8142) as f_in:
            for line in f_in:
                line = line.strip("\n").strip().split(",")
                lines += 1
                if lines <= 465685:
                    continue
                length = len(line)
                req = predict_pb2.PredictRequest()
                req.model_spec.name = self.model_name
                req.inputs["inputs"].CopyFrom(tf.make_tensor_proto([line], shape=[1, length]))
                req.inputs["inputs_length"].CopyFrom(tf.make_tensor_proto([length], shape=[1]))
                response = self.stub.Predict.future(req, 10)
                res = response.result()
                outputs = tf.make_ndarray(res.outputs["predict_tags"])[0]
                outputs = [b.decode("utf-8") for b in outputs]
                results = self._translate_tag_to_string(list(line), outputs)
                print("\nlines:%s\n results:%s" % (str(lines), results))
                file = r"D:\PyCharmProjects\deepseg\data\model_results_all.txt"
                with open(file, mode="a+", encoding="utf8", buffering=8192) as f:
                    f.write(results + "\n")
            #     vocabs = results.split(" ")
            #     for vs in vocabs:
            #         if len(vs) < 2:
            #             continue
            #         if self.pattern.findall(vs):
            #             continue
            #         counter[vs] += 1
            # words = counter.most_common(250000)
            # vocabs = set()
            # for item in words:
            #     vocabs.add(item[0])
            # self._write_vocab(vocabs=vocabs, file=self.output_file)

    # def serving_more(self, part):
    #     lines = 0
    #     part_out = part + ".out"
    #     with open(part, 'rt', encoding='utf8', buffering=8142)as fin, \
    #             open(part_out, 'wt', encoding='utf8', buffering=8142)as fout:
    #         for line in fin:
    #             line = line.strip("\n").strip().split(",")
    #             leng = len(line)
    #             req = predict_pb2.PredictRequest()
    #             req.model_spec.name = self.model_name
    #             req.inputs["inputs"].CopyFrom(tf.make_tensor_proto(line), shape=[1, leng])
    #             req.inputs["inputs_length"].CopyFrom(tf.make_tensor_proto([leng]), shape=[1])
    #             response = self.stub.Predict.future(req, 10)
    #             rsp = response.result()
    #             outputs = tf.make_ndarray(rsp.outputs["predict_tags"][0])
    #             outputs = [b.decode("utf8") for b in outputs]
    #             results = self._translate_tag_to_string(list(line), outputs)
    #             lines += 1
    #             print("\n lines:%s\n results:%s" % (str(lines), results))
    #             fout.write(results + "\n")
    #     return part_out
