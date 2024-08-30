#
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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from singa import device
from singa import opt
from singa import tensor

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from model import gan_mlp
from utils import load_data
from utils import print_log


class VANILLA():

    def __init__(self,
                 dev,
                 rows=28,
                 cols=28,
                 channels=1,
                 noise_size=100,
                 hidden_size=128,
                 batch=128,
                 interval=1000,
                 learning_rate=0.001,
                 iterations=1000000,
                 dataset_filepath='mnist.pkl.gz',
                 file_dir='vanilla_images/'):
        self.dev = dev
        self.rows = rows
        self.cols = cols
        self.channels = channels
        self.feature_size = self.rows * self.cols * self.channels
        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.batch = batch
        self.batch_size = self.batch // 2
        self.interval = interval
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.dataset_filepath = dataset_filepath
        self.file_dir = file_dir
        self.model = gan_mlp.create_model(noise_size=self.noise_size,
                                          feature_size=self.feature_size,
                                          hidden_size=self.hidden_size)

    def train(self):
        train_data, _, _, _, _, _ = load_data(self.dataset_filepath)
        dev = device.create_cuda_gpu_on(0)
        dev.SetRandSeed(0)
        np.random.seed(0)

        # sgd = opt.SGD(lr=self.learning_rate, momentum=0.9, weight_decay=1e-5)
        sgd = opt.Adam(lr=self.learning_rate)

        noise = tensor.Tensor((self.batch_size, self.noise_size), dev,
                              tensor.float32)
        real_images = tensor.Tensor((self.batch_size, self.feature_size), dev,
                                    tensor.float32)
        real_labels = tensor.Tensor((self.batch_size, 1), dev, tensor.float32)
        fake_labels = tensor.Tensor((self.batch_size, 1), dev, tensor.float32)

        # attached model to graph
        self.model.set_optimizer(sgd)
        self.model.compile([noise],
                           is_train=True,
                           use_graph=False,
                           sequential=True)

        real_labels.set_value(1.0)
        fake_labels.set_value(0.0)

        for iteration in range(self.iterations):
            idx = np.random.randint(0, train_data.shape[0], self.batch_size)
            real_images.copy_from_numpy(train_data[idx])

            self.model.train()

            # Training the Discriminative Net
            _, d_loss_real = self.model.train_one_batch_dis(
                real_images, real_labels)

            noise.uniform(-1, 1)
            fake_images = self.model.forward_gen(noise)
            _, d_loss_fake = self.model.train_one_batch_dis(
                fake_images, fake_labels)

            d_loss = tensor.to_numpy(d_loss_real)[0] + tensor.to_numpy(
                d_loss_fake)[0]

            # Training the Generative Net
            noise.uniform(-1, 1)
            _, g_loss_tensor = self.model.train_one_batch(noise, real_labels)

            g_loss = tensor.to_numpy(g_loss_tensor)[0]

            if iteration % self.interval == 0:
                self.model.eval()
                self.save_image(iteration)
                print_log(' The {} iteration, G_LOSS: {}, D_LOSS: {}'.format(
                    iteration, g_loss, d_loss))

    def save_image(self, iteration):
        demo_row = 5
        demo_col = 5
        if not hasattr(self, "demo_noise"):
            self.demo_noise = tensor.Tensor(
                (demo_col * demo_row, self.noise_size), dev, tensor.float32)
        self.demo_noise.uniform(-1, 1)
        gen_imgs = self.model.forward_gen(self.demo_noise)
        gen_imgs = tensor.to_numpy(gen_imgs)
        show_imgs = np.reshape(
            gen_imgs, (gen_imgs.shape[0], self.rows, self.cols, self.channels))
        fig, axs = plt.subplots(demo_row, demo_col)
        cnt = 0
        for r in range(demo_row):
            for c in range(demo_col):
                axs[r, c].imshow(show_imgs[cnt, :, :, 0], cmap='gray')
                axs[r, c].axis('off')
                cnt += 1
        fig.savefig("{}{}.png".format(self.file_dir, iteration))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN over MNIST')
    parser.add_argument('filepath', type=str, help='the dataset path')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        print('Using GPU')
        dev = device.create_cuda_gpu()
    else:
        print('Using CPU')
        dev = device.get_default_device()

    if not os.path.exists('vanilla_images/'):
        os.makedirs('vanilla_images/')

    rows = 28
    cols = 28
    channels = 1
    noise_size = 100
    hidden_size = 128
    batch = 128
    interval = 1000
    learning_rate = 0.0005
    iterations = 1000000
    dataset_filepath = 'mnist.pkl.gz'
    file_dir = 'vanilla_images/'
    vanilla = VANILLA(dev, rows, cols, channels, noise_size, hidden_size, batch,
                      interval, learning_rate, iterations, dataset_filepath,
                      file_dir)
    vanilla.train()
