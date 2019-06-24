import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
import os

channel = implementations.insecure_channel("10.100.51.111", 8877)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
model_name = "deepseg_test"


def serving(source):
    inputs = source
    inputs_length = len(source)
    req = predict_pb2.PredictRequest()
    req.model_spec.name = model_name
    req.inputs["inputs"].CopyFrom(tf.make_tensor_proto([source], shape=[1, inputs_length]))
    req.inputs["inputs_length"].CopyFrom(tf.make_tensor_proto([inputs_length], shape=[1]))
    response = stub.Predict.future(req, 10)
    res = response.result()
    outputs = tf.make_ndarray(res.outputs["predict_tags"])[0]
    outputs = [b.decode("utf-8") for b in outputs]
    return translate(inputs, outputs)


def translate(src_list, tag_list):
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
    return True, "".join(result).strip()


def translate_more(src_lists, tag_lists):
    if not src_lists or not tag_lists:
        raise Exception("Source lists or Tag lists is None !")
    if len(src_lists) != len(tag_lists):
        raise Exception(" The length of source lists is not equal to tag lists !")
    result = []
    for src_list, tag_list in zip(src_lists, tag_lists):
        _b, _s = translate(src_list, tag_list)
        result.append(_s)
    return result


def testify(feature, label, output_file):
    same_lines = 0
    diff_lines = 0
    with open(feature, mode="r+", encoding="utf-8") as features, \
            open(label, mode="r+", encoding="utf-8") as labels, \
            open(output_file, mode="w+", encoding="utf-8") as fout:
        for f, l in zip(features, labels):
            _source = f.replace("\n", "").strip().split(",")
            _tags = l.replace("\n", "").strip().split(",")
            b_std, std = translate(_source, _tags)
            if not b_std:
                continue
            b_seg, seg = serving(_source)
            if not b_seg:
                continue
            if std == seg:
                same_lines += 1
            else:
                diff_lines += 1
                fout.write("\nSTD:" + std + "\n")
                fout.write("SEG:" + seg + "\n")
                print("STD:" + std)
                print("SEG:" + seg)
    print("diff rate %.4f" % (diff_lines / (diff_lines + same_lines)))


if __name__ == '__main__':
    _src = ["上", "海", "市", "浦", "东", "新", "区", "张", "东", "路", "1387", "号"]
    # _res = serving(_src)
    # print(_res)
    proj_dir = os.path.dirname(os.path.dirname(__file__))
    test_feature = "data/test.feature.txt"
    test_label = "data/test.label.txt"
    _output_file = "data/test.seg.txt"
    test_feature = os.path.join(proj_dir, test_feature)
    test_label = os.path.join(proj_dir, test_label)
    _output_file = os.path.join(proj_dir,_output_file)
    testify(test_feature, test_label, _output_file)
