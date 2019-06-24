"""Example of a translation client."""
from __future__ import print_function
import argparse
import tensorflow as tf
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


def pad_batch(batch_tokens):
    lengths = [len(tokens) for tokens in batch_tokens]
    max_length = max(lengths)
    for tokens, length in zip(batch_tokens, lengths):
        if max_length > length:
            tokens += [""] * (max_length - length)
    return batch_tokens, lengths, max_length


def extract_prediction(result):
    batch_lengths = tf.make_ndarray(result.outputs["length"])
    batch_predictions = tf.make_ndarray(result.outputs["tokens"])
    for hypotheses, lengths in zip(batch_predictions, batch_lengths):
        # Only consider the first hypothesis (the best one).
        best_hypothesis = hypotheses[0]
        best_length = lengths[0] - 1  # Ignore </s>
        yield best_hypothesis[:best_length]


def send_request(stub, model_name, batch_tokens, timeout=5.0):
    batch_tokens, lengths, max_length = pad_batch(batch_tokens)
    batch_size = len(lengths)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs["tokens"].CopyFrom(
        tf.make_tensor_proto(batch_tokens, shape=(batch_size, max_length)))
    request.inputs["length"].CopyFrom(
        tf.make_tensor_proto(lengths, shape=(batch_size,)))
    return stub.Predict.future(request, timeout)


def translate(stub, model_name, batch_text, timeout=5.0):
    batch_input = batch_text
    future = send_request(stub, model_name, batch_input, timeout=timeout)
    result = future.result()
    batch_output = extract_prediction(result)
    return batch_output


def main():
    parser = argparse.ArgumentParser(description="Translation client example")
    parser.add_argument("--model_name", default="ende",
                        help="model name")
    parser.add_argument("--host", default="10.100.51.111",
                        help="model server host")
    parser.add_argument("--port", type=int, default=8090,
                        help="model server port")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="request timeout")
    args = parser.parse_args()

    channel = grpc.insecure_channel("%s:%d" % (args.host, args.port))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    batch_input = [["Hello", "world", "!"],
                   ["My", "name", "is", "John", "."],
                   ["I", "live", "on", "the", "West", "coast", "."]]
    # batch_input = ["Hello world!", "My name is John.", "I live on the West coast."]
    batch_output = translate(stub, args.model_name, batch_input, timeout=args.timeout)
    print(batch_output)
    for input_text, output_text in zip(batch_input, batch_output):
        print("{} ||| {}".format(input_text, output_text))


if __name__ == "__main__":
    main()
