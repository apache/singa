# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import time
import numpy as np
import traceback
from argparse import ArgumentParser
from scipy.misc import imread

from singa import device
from singa import tensor
from singa import image_tool

from rafiki.agent import Agent, MsgType
import model

tool = image_tool.ImageTool()
num_augmentation = 1
crop_size = 224
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in \
        ["PNG", "png", "jpg", "JPG", "JPEG", "jpeg"]


def serve(net, label_map, dev, agent, topk=5):
    '''Serve to predict image labels.
    It prints the topk food names for each image.
    Args:
        label_map: a list of food names, corresponding to the index in meta_file
    '''

    images = tensor.Tensor((num_augmentation, 3, crop_size, crop_size), dev)
    while True:
        msg, val = agent.pull()
        if msg is None:
            time.sleep(0.1)
            continue
        msg = MsgType.parse(msg)
        if msg.is_request():
            try:
                # process images
                img = imread(val['image'], mode='RGB').astype(np.float32) / 255
                height,width = img.shape[:2]
                img -= mean
                img /= std
                img = img.transpose((2, 0, 1))
                img = img[:, (height-224)//2:(height+224)//2,
                          (width-224)//2:(width+224)//2]
                images.copy_from_numpy(img)
                print("input: ", images.l1())
                # do prediction
                y = net.predict(images)
                prob = np.average(tensor.to_numpy(y), 0)
                idx = np.argsort(-prob)
                # prepare results
                response = ""
                for i in range(topk):
                    response += "%s:%f <br/>" % (label_map[idx[i]],
                                                 prob[idx[i]])
            except Exception:
                traceback.print_exc()
                response = "Sorry, system error during prediction."
            except SystemExit:
                traceback.print_exc()
                response = "Sorry, error triggered sys.exit() during prediction."
            agent.push(MsgType.kResponse, response)
        elif msg.is_command():
            if MsgType.kCommandStop.equal(msg):
                print('get stop command')
                agent.push(MsgType.kStatus, "success")
                break
            else:
                print('get unsupported command %s' % str(msg))
                agent.push(MsgType.kStatus, "Unknown command")
        else:
            print('get unsupported message %s' % str(msg))
            agent.push(MsgType.kStatus, "unsupported msg; going to shutdown")
            break
    print("server stop")


def main():
    try:
        # Setup argument parser
        parser = ArgumentParser(description='DenseNet inference')

        parser.add_argument("--port", default=9999, help="listen port")
        parser.add_argument("--use_cpu", action="store_true",
                            help="If set, load models onto CPU devices")
        parser.add_argument("--parameter_file", default="densenet-121.pickle")
        parser.add_argument("--depth", type=int, choices=[121, 169, 201, 161],
                            default=121)

        parser.add_argument('--nb_classes', default=1000, type=int)

        # Process arguments
        args = parser.parse_args()
        port = args.port

        # start to train
        agent = Agent(port)

        net = model.create_net(args.depth, args.nb_classes, 0, args.use_cpu)
        if args.use_cpu:
            print('Using CPU')
            dev = device.get_default_device()
        else:
            print('Using GPU')
            dev = device.create_cuda_gpu()
            net.to_device(dev)
        print('start to load parameter_file')
        model.init_params(net, args.parameter_file)
        print('Finish loading models')

        labels = np.loadtxt('synset_words.txt', str, delimiter='\t ')
        serve(net, labels, dev, agent)
        # wait the agent finish handling http request
        agent.stop()

    except SystemExit:
        return
    except Exception:
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


if __name__ == '__main__':
    main()