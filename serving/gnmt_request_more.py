import tensorflow as tf

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

model_name = "address"
host = "10.100.51.111"
port = 8090
timeout = 10.0

channel = grpc.insecure_channel("%s:%d" % (host, port))
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def send_request(batch_tokens):
    batch_tokens, lengths, max_length = pad_batch(batch_tokens)
    batch_size = len(lengths)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs["tokens"].CopyFrom(
        tf.make_tensor_proto([batch_tokens], shape=(batch_size, max_length)))
    request.inputs["length"].CopyFrom(
        tf.make_tensor_proto(lengths, shape=(batch_size,)))

    future = stub.Predict.future(request, timeout)
    batch_output = parse_translation_result(future.result())
    batch_output = parse_result_to_utf8_text(batch_output)
    return batch_output


def parse_translation_result(result):
    batch_predictions = tf.make_ndarray(result.outputs["tokens"])
    batch_lengths = tf.make_ndarray(result.outputs["length"])

    for hypotheses, lengths in zip(batch_predictions, batch_lengths):
        best_hypothesis = hypotheses[0]
        best_length = lengths[0] - 1
        yield best_hypothesis[:best_length]


def parse_result_to_utf8_text(results):
    batch_output = []
    for result in results:
        tokens = []
        for r in list(result):
            tokens.append(str(r, encoding='utf8'))
        batch_output.append(tokens)
    return batch_output


def pad_batch(batch_tokens):
    lengths = [len(address) for address in batch_tokens]
    max_lengths = max(lengths)
    for tokens, length in zip(batch_tokens, lengths):
        if max_lengths > length:
            tokens += [""] * (max_lengths - length)

    return batch_tokens, lengths, max_lengths


if __name__ == "__main__":
    # address = ['土', '悔', '市', '浦', '东', '新', '区', '张', '栋', '路', '1387', '号']
    addresses = [['土悔市', '浦东新区', '张栋路', '1387', '号'],
               ['土悔市', '浦东新区', '章东路'],
               ['土悔市', '浦A新区']]

    result = send_request(addresses)
    print(result)
