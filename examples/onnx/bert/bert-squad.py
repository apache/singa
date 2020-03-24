import os
import urllib.request
import gzip
import zipfile
import numpy as np
import codecs
import tarfile
import warnings
import glob
import json

from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa import sonnx
import onnx
from onnx import version_converter, helper, numpy_helper, shape_inference
from onnx.checker import check_model
import onnx.utils
import tokenization
from run_onnx_squad import read_squad_examples, convert_examples_to_features, RawResult, write_predictions

max_answer_length = 30
max_seq_length = 256
doc_stride = 128
max_query_length = 64
n_best_size = 20
batch_size = 1


def load_model():
    url = 'https://media.githubusercontent.com/media/onnx/models/master/text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz'
    download_dir = '/tmp/'
    filename = os.path.join(download_dir, 'download_sample_10', '.',
                            'bertsquad10.onnx')
    with tarfile.open(check_exist_or_download(url), 'r') as t:
        t.extractall(path=download_dir)
    return filename


def load_vocab():
    url = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
    download_dir = '/tmp/'
    filename = os.path.join(download_dir, 'uncased_L-12_H-768_A-12', '.',
                            'vocab.txt')
    with zipfile.ZipFile(check_exist_or_download(url), 'r') as z:
        z.extractall(path=download_dir)
    return filename


def load_dataset(test_data_dir):
    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        onnx_tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            onnx_tensor.ParseFromString(f.read())
        np_tensor = numpy_helper.to_array(onnx_tensor)
        if np_tensor.dtype == "int64":
            np_tensor = np_tensor.astype(np.int32)
        inputs.append(np_tensor)

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        onnx_tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            onnx_tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(onnx_tensor))
    return inputs, ref_outputs


def check_exist_or_download(url):
    download_dir = '/tmp/'
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename


def update_batch_size(onnx_model):
    for model_input in onnx_model.graph.input[0:4]:
        model_input.type.tensor_type.shape.dim[0].dim_value = batch_size
    for model_output in onnx_model.graph.output[0:3]:
        model_output.type.tensor_type.shape.dim[0].dim_value = batch_size
    inferred_model = shape_inference.infer_shapes(onnx_model)
    return inferred_model


class Infer:

    def __init__(self, sg_ir):
        self.sg_ir = sg_ir

    def forward(self, x):
        return sg_ir.run(x)


def preprocessing():
    vocab_file = load_vocab()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=True)
    predict_file = os.path.join(os.path.dirname(__file__), 'inputs.json')
    # print content
    with open(predict_file) as json_file:
        test_data = json.load(json_file)
        print(json.dumps(test_data, indent=2))

    eval_examples = read_squad_examples(input_file=predict_file)

    # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
    input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
        eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length)
    return input_ids, input_mask, segment_ids, extra_data, eval_examples


def postprocessing(eval_examples, extra_data, all_results):
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    write_predictions(eval_examples, extra_data, all_results, n_best_size,
                      max_answer_length, True, output_prediction_file,
                      output_nbest_file)

    # print results
    with open(output_prediction_file) as json_file:
        test_data = json.load(json_file)
        print(json.dumps(test_data, indent=2))


if __name__ == "__main__":
    dev = device.create_cuda_gpu()
    model_path = load_model()
    onnx_model = onnx.load(model_path)
    onnx_model = update_batch_size(onnx_model)
    autograd.training = False

    # preprocessing
    input_ids, input_mask, segment_ids, extra_data, eval_examples = preprocessing(
    )

    sg_ir = None
    n = len(input_ids)
    bs = batch_size
    all_results = []

    tmp_dict = {}
    for idx in range(0, n):
        item = eval_examples[idx]
        inputs = [
            np.array([item.qas_id], dtype=np.int32),
            segment_ids[idx:idx + bs].astype(np.int32),
            input_mask[idx:idx + bs].astype(np.int32),
            input_ids[idx:idx + bs].astype(np.int32),
        ]

        if sg_ir is None:
            sg_ir = sonnx.prepare(onnx_model,
                                  device=dev,
                                  init_inputs=inputs,
                                  keep_initializers_as_inputs=False)
            model = Infer(sg_ir)

        x_batch = []
        for inp in inputs:
            tmp_tensor = tensor.from_numpy(inp)
            tmp_tensor.to_device(dev)
            x_batch.append(tmp_tensor)
        outputs = model.forward(x_batch)
        result = []
        for outp in outputs:
            result.append(tensor.to_numpy(outp))

        in_batch = result[1].shape[0]
        start_logits = [float(x) for x in result[1][0].flat]
        end_logits = [float(x) for x in result[0][0].flat]
        for i in range(0, in_batch):
            unique_id = len(all_results)
            all_results.append(
                RawResult(unique_id=unique_id,
                          start_logits=start_logits,
                          end_logits=end_logits))
    # postprocessing
    postprocessing(eval_examples, extra_data, all_results)