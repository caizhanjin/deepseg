import tensorflow as tf

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

model_name = "address"
host = "10.100.51.111"
port = 8090
timeout = 10.0

channel = grpc.insecure_channel("%s:%d" % (host, port))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def send_request(tokens):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs["tokens"].CopyFrom(
        tf.make_tensor_proto([tokens], shape=(1, len(tokens))))
    request.inputs["length"].CopyFrom(
        tf.make_tensor_proto([len(tokens)], shape=(1,)))

    future = stub.Predict.future(request, timeout)
    result = parse_translation_result(future.result())
    result = parse_result_to_utf8_text(result)

    return result


def parse_translation_result(result):
    hypotheses = tf.make_ndarray(result.outputs["tokens"])[0]
    lengths = tf.make_ndarray(result.outputs["length"])[0]

    best_hypothesis = hypotheses[0]
    best_length = lengths[0]

    return best_hypothesis[0:best_length-1]


def parse_result_to_utf8_text(result):
    tokens = []
    for r in list(result):
        tokens.append(str(r, encoding='utf8'))
    return " ".join(tokens)


if __name__ == "__main__":
    # address = ['土', '悔', '市', '浦', '东', '新', '区', '张', '栋', '路', '1387', '号']
    address = ['土悔市', '浦东新区', '张栋路', '1387', '号']
    result = send_request(address)
    print(result)
