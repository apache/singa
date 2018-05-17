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
# =============================================================================
from singa import device
from singa import tensor
from singa import image_tool
from singa import layer
from rafiki.agent import Agent, MsgType

import sys
import time
import traceback
from argparse import ArgumentParser
import numpy as np

import inception_v4
import inception_v3


def serve(agent, net, use_cpu, parameter_file, topk=5):
    if use_cpu:
        print('running with cpu')
        dev = device.get_default_device()
        layer.engine = 'singacpp'
    else:
        print("runing with gpu")
        dev = device.create_cuda_gpu()
    agent = agent

    print('Start intialization............')
    # fix the bug when creating net
    if net == 'v3':
        model = inception_v3
    else:
        model = inception_v4
    net, _ = model.create_net(is_training=False)
    net.load(parameter_file, use_pickle=True)
    net.to_device(dev)
    print('End intialization............')

    labels = np.loadtxt('synset_words.txt', str, delimiter='\t').tolist()
    labels.insert(0, 'empty background')
    while True:
        key, val = agent.pull()
        if key is None:
            time.sleep(0.1)
            continue
        msg_type = MsgType.parse(key)
        if msg_type.is_request():
            try:
                response = ""
                ratio = 0.875
                img = image_tool.load_img(val['image'])
                height, width = img.size[0], img.size[1]
                print(img.size)
                crop_h, crop_w = int(height * ratio), int(width * ratio)
                img = np.array(image_tool.crop(img,\
                      (crop_h, crop_w), 'center').\
                      resize((299, 299))).astype(np.float32) / float(255)
                img -= 0.5
                img *= 2
                # img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
                img = img.transpose((2, 0, 1))
                images = np.expand_dims(img, axis=0)
                x = tensor.from_numpy(images.astype(np.float32))
                x.to_device(dev)
                y = net.predict(x)
                prob = np.average(tensor.to_numpy(y), 0)
                # sort and reverse
                idx = np.argsort(-prob)[0:topk]
                for i in idx:
                    response += "%s:%s<br/>" % (labels[i], prob[i])
            except:
                traceback.print_exc()
                response = "Sorry, system error during prediction."
            agent.push(MsgType.kResponse, response)
        elif MsgType.kCommandStop.equal(msg_type):
                print('get stop command')
                agent.push(MsgType.kStatus, "success")
                break
        else:
            print('get unsupported message %s' % str(msg_type))
            agent.push(MsgType.kStatus, "Unknown command")
            break
        # while loop
    print("server stop")


def main():
    try:
        # Setup argument parser
        parser = ArgumentParser(description=\
                                "InceptionV3 and V4 for image classification")
        parser.add_argument("--model", choices=['v3', 'v4'], default='v4')
        parser.add_argument("-p", "--port", default=9999, help="listen port")
        parser.add_argument("-C", "--use_cpu", action="store_true")
        parser.add_argument("--parameter_file", default="inception_v4.pickle",
                            help="relative path")

        # Process arguments
        args = parser.parse_args()
        port = args.port

        # start to train
        agent = Agent(port)
        serve(agent, args.model, args.use_cpu, args.parameter_file)
        agent.stop()

    except SystemExit:
        return
    except:
        traceback.print_exc()
        sys.stderr.write("  for help use --help \n\n")
        return 2


if __name__ == '__main__':
    main()
