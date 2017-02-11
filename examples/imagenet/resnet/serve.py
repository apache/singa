import os
import sys
import time
import numpy as np
import threading
import traceback
from scipy.misc import imread, imresize
from argparse import ArgumentParser

from singa import device
from singa import tensor
from singa import data
from singa import image_tool
from singa import metric
from rafiki.agent import Agent, MsgType
import model

tool = image_tool.ImageTool()
num_augmentation = 10
crop_size = 224
mean = np.array([0.485, 0.456, 0.406])
std = np.array([ 0.229, 0.224, 0.225])
def image_transform(img):
    '''Input an image path and return a set of augmented images (type Image)'''
    global tool
    return tool.load(img).resize_by_list([256]).crop5((crop_size, crop_size), 5).flip(2).get()


def predict(net, images, num=10):
    '''predict probability distribution for one net.

    Args:
        net: neural net (vgg or resnet)
        images: a batch of augmented images (type numpy)
        num: num of augmentations
    '''
    prob = net.predict(images)
    prob = tensor.to_numpy(prob)
    prob = prob.reshape((images.shape[0] / num, num, -1))
    prob = np.average(prob, 1)
    return prob


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in \
        ["PNG", "png", "jpg", "JPG", "JPEG", "jpeg"]


def serve(net, label_map, dev, agent, topk=5):
    '''Serve to predict image labels.

    It prints the topk food names for each image.

    Args:
        label_map: a list of food names, corresponding to the index in meta_file
    '''

    images =tensor.Tensor((num_augmentation, 3, crop_size, crop_size), dev)
    while True:
        msg, val = agent.pull()
        if msg is None:
            time.sleep(0.1)
            continue
        msg = MsgType.parse(msg)
        if msg.is_request():
            try:
                # process images
                im = [np.array(x.convert('RGB'), dtype=np.float32).transpose(2, 0, 1) for x in image_transform(val['image'])]
                im = np.array(im) / 256
                im -= mean[np.newaxis, :, np.newaxis, np.newaxis]
                im /= std[np.newaxis, :, np.newaxis, np.newaxis]
                images.copy_from_numpy(im)
                print "input: ", images.l1()
                # do prediction
                prob = predict(net, images, num_augmentation)[0]
                idx = np.argsort(-prob)
                # prepare results
                response = ""
                for i in range(topk):
                    response += "%s:%f <br/>" % (label_map[idx[i]], prob[idx[i]])
            except:
                traceback.print_exc()
                response = "sorry, system error during prediction."
            agent.push(MsgType.kResponse, response)
        elif msg.is_command():
            if MsgType.kCommandStop.equal(msg):
                print 'get stop command'
                agent.push(MsgType.kStatus, "success")
                break
            else:
                print 'get unsupported command %s' % str(msg)
                agent.push(MsgType.kStatus, "Unknown command")
        else:
            print 'get unsupported message %s' % str(msg)
            agent.push(MsgType.kStatus, "unsupported msg; going to shutdown")
            break
    print "server stop"

def main():
    try:
        # Setup argument parser
        parser = ArgumentParser(description="Wide residual network")

        parser.add_argument("-p", "--port", default=9999, help="listen port")
        parser.add_argument("-c", "--use_cpu", action="store_true",
                            help="If set, load models onto CPU devices")
        parser.add_argument("--parameter_file", default="wrn-50-2.pickle")
        parser.add_argument("--model", choices = ['resnet', 'wrn', 'preact', 'addbn'], default='wrn')
        parser.add_argument("--depth", type=int, choices = [18, 34, 50, 101, 152, 200], default='50')

        # Process arguments
        args = parser.parse_args()
        port = args.port

        # start to train
        agent = Agent(port)

        net = model.create_net(args.model, args.depth)
        dev = device.create_cuda_gpu()
        net.to_device(dev)
        model.init_params(net, args.parameter_file)
        print 'Finish loading models'

        labels = np.loadtxt('synset_words.txt', str, delimiter='\t ')
        serve(net, labels, dev, agent)

        # acc = evaluate(net, '../val_list.txt',  'image/val', dev)
        # print acc

        # wait the agent finish handling http request
        agent.stop()
    except SystemExit:
        return
    except:
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


if __name__ == '__main__':
    main()
