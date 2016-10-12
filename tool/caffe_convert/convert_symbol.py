from google.protobuf import text_format
import argparse, re, sys
import caffe_parse.caffe_pb2

def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    if not file:
        raise self.ProcessException("ERROR (" + filepath + ")!")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object

def readProtoNetParameterFile(filepath):
    netparameter_config = caffe_parse.caffe_pb2.NetParameter()
    return readProtoFile(filepath, netparameter_config)

def proto2script(proto_file):
    proto = readProtoNetParameterFile(proto_file)
    connection = dict()
    symbols = dict()
    top = dict()
    flatten_count = 0
    symbol_string = ""
    layer = ''
    if len(proto.layer):
        layer = proto.layer
    elif len(proto.layers):
        layer = proto.layers
    else:
        raise Exception('Invalid proto file.')
    # set default input size to network
    first_conv = True
    first_dense = True
    input_dim = [3,224,224]
    input_name = layer[0].bottom[0]
    output_name = ""
    mapping = {input_name : 'data'}
    need_flatten = {input_name : False}
    transpose = False
    for i in range(len(layer)):
        type_string = ''
        param_string = ''
        name = re.sub('[-/]', '_', layer[i].name)
        if layer[i].type == 'Convolution' or layer[i].type == 4:
            type_string = 'Conv2D'
            param = layer[i].convolution_param
            if isinstance(param.pad, int):
                pad = param.pad
            else:
                pad = 0 if len(param.pad) == 0 else param.pad[0]
            if isinstance(param.kernel_size, int):
                kernel_size = param.kernel_size
            else:
                kernel_size = param.kernel_size[0]
            if isinstance(param.stride, int):
                stride = param.stride
            else:
                stride = 1 if len(param.stride) == 0 else param.stride[0]
            if first_conv:
                param_string = " '%s', %d, kernel=%d, stride=%d, use_bias=%s, pad=%d, input_sample_shape=%s" %\
                (name, param.num_output, kernel_size, stride, param.bias_term, pad, input_dim)
            else:
                param_string = " '%s', %d, kernel=%d, stride=%d, use_bias=%s, pad=%d" %\
                (name, param.num_output, kernel_size, stride, param.bias_term, pad)
            first_conv = False
        if layer[i].type == 'Pooling' or layer[i].type == 17:
            param = layer[i].pooling_param
            param_string = " '%s', kernel=%d, stride=%d, pad=%d" % (name, param.kernel_size, param.stride, param.pad)
            if param.pool == 0:
                type_string = 'MaxPooling2D'
            elif param.pool == 1:
                type_string = 'AvgPooling2D'
            else:
                raise Exception("Unknown Pooling Method!")
        if layer[i].type == 'ReLU' or layer[i].type == 18:
            type_string = 'Activation'
            param_string = " '%s', mode='%s' " % (name, 'relu')
        if layer[i].type == 'InnerProduct' or layer[i].type == 14:
            type_string = 'Dense'
            param = layer[i].inner_product_param
            param_string = " '%s', %d, use_bias=%s " % (name, param.num_output, param.bias_term)
        if layer[i].type == 'Dropout' or layer[i].type == 6:
            type_string = 'Dropout'
            param = layer[i].dropout_param
            param_string = " '%s', p=%f" % (name, param.dropout_ratio)
        if layer[i].type == 'Softmax' or layer[i].type == 20:
            type_string = 'Softmax'
            param_string = " '%s' " % (name)
        if layer[i].type == 'Flatten' or layer[i].type == 8:
            type_string = 'Flatten'
            param_string = " '%s' " % (name)
        if type_string == '':
            raise Exception('Unknown Layer %s!' % layer[i].type)
        bottom = layer[i].bottom
        if len(bottom) == 1 and type_string!="Softmax":
            symbol_string += "%s = net.add(layer.%s( %s ) )\n" % \
                (name, type_string, param_string)
        elif type_string!="Softmax":
            symbol_string += "%s = net.add(layer.%s( *[%s] %s) )\n" %\
                (name, type_string, ','.join([mapping[x] for x in bottom]), param_string)
        for j in range(len(layer[i].top)):
            mapping[layer[i].top[j]] = name
        output_name = name
    return symbol_string, output_name, input_dim

def proto2symbol(proto_file):
    sym, output_name, input_dim = proto2script(proto_file)
    print "transformed Singa VGG-16 net is:\n", sym
    sym = "from singa import layer \n" \
        +"from singa import initializer \n" \
        +"from singa import metric\n" \
        +"from singa import loss\n" \
        +"from singa import net as ffnet\n" \
        +"net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())\n" \
        +sym
    exec(sym)
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
    args = parser.parse_args()
    symbol_string, output_name, input_dim = proto2script(args.caffe_prototxt)
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'w') as fout:
            fout.write(symbol_string)
    else:
        print(symbol_string)

if __name__ == '__main__':
    main()
