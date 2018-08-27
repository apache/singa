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
from singa import initializer
from singa import layer
from singa import loss
from singa import net as ffnet
from singa import optimizer
from singa import tensor

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import load_data
from utils import print_log

class LSGAN():
	def  __init__(self, dev, rows=28, cols=28, channels=1, noise_size=100, hidden_size=128, batch=128, 
		interval=1000, learning_rate=0.001, epochs=1000000, d_steps=3, g_steps=1, 
		dataset_filepath='mnist.pkl.gz', file_dir='lsgan_images/'):
		self.dev = dev
		self.rows = rows
		self.cols = cols
		self.channels = channels
		self.feature_size = self.rows * self.cols * self.channels
		self.noise_size = noise_size
		self.hidden_size = hidden_size
		self.batch = batch
		self.batch_size = self.batch//2
		self.interval = interval
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.d_steps = d_steps
		self.g_steps = g_steps
		self.dataset_filepath = dataset_filepath
		self.file_dir = file_dir

		self.g_w0_specs = {'init': 'xavier',}
		self.g_b0_specs = {'init': 'constant', 'value': 0,}
		self.g_w1_specs = {'init': 'xavier',}
		self.g_b1_specs = {'init': 'constant', 'value': 0,}
		self.gen_net = ffnet.FeedForwardNet(loss.SquaredError(),)
		self.gen_net_fc_0 = layer.Dense(name='g_fc_0', num_output=self.hidden_size, use_bias=True, 
			W_specs=self.g_w0_specs, b_specs=self.g_b0_specs, input_sample_shape=(self.noise_size,))
		self.gen_net_relu_0 = layer.Activation(name='g_relu_0', mode='relu',input_sample_shape=(self.hidden_size,))
		self.gen_net_fc_1 = layer.Dense(name='g_fc_1', num_output=self.feature_size, use_bias=True, 
			W_specs=self.g_w1_specs, b_specs=self.g_b1_specs, input_sample_shape=(self.hidden_size,))
		self.gen_net_sigmoid_1 = layer.Activation(name='g_relu_1', mode='sigmoid', input_sample_shape=(self.feature_size,))
		self.gen_net.add(self.gen_net_fc_0)
		self.gen_net.add(self.gen_net_relu_0)
		self.gen_net.add(self.gen_net_fc_1)
		self.gen_net.add(self.gen_net_sigmoid_1)
		for (p, specs) in zip(self.gen_net.param_values(), self.gen_net.param_specs()):
			filler = specs.filler
			if filler.type == 'gaussian':
				p.gaussian(filler.mean, filler.std)
			elif filler.type == 'xavier':
				initializer.xavier(p)
			else: 
				p.set_value(0)
			print(specs.name, filler.type, p.l1())	
		self.gen_net.to_device(self.dev)		

		self.d_w0_specs = {'init': 'xavier',}
		self.d_b0_specs = {'init': 'constant', 'value': 0,}
		self.d_w1_specs = {'init': 'xavier',}
		self.d_b1_specs = {'init': 'constant', 'value': 0,}			
		self.dis_net = ffnet.FeedForwardNet(loss.SquaredError(),)
		self.dis_net_fc_0 = layer.Dense(name='d_fc_0', num_output=self.hidden_size, use_bias=True, 
			W_specs=self.d_w0_specs, b_specs=self.d_b0_specs, input_sample_shape=(self.feature_size,))
		self.dis_net_relu_0 = layer.Activation(name='d_relu_0', mode='relu',input_sample_shape=(self.hidden_size,))
		self.dis_net_fc_1 = layer.Dense(name='d_fc_1', num_output=1,  use_bias=True, 
			W_specs=self.d_w1_specs, b_specs=self.d_b1_specs, input_sample_shape=(self.hidden_size,))
		self.dis_net.add(self.dis_net_fc_0)
		self.dis_net.add(self.dis_net_relu_0)
		self.dis_net.add(self.dis_net_fc_1)			
		for (p, specs) in zip(self.dis_net.param_values(), self.dis_net.param_specs()):
			filler = specs.filler
			if filler.type == 'gaussian':
				p.gaussian(filler.mean, filler.std)
			elif filler.type == 'xavier':
				initializer.xavier(p)
			else: 
				p.set_value(0)
			print(specs.name, filler.type, p.l1())
		self.dis_net.to_device(self.dev)

		self.combined_net = ffnet.FeedForwardNet(loss.SquaredError(), )
		for l in self.gen_net.layers:
			self.combined_net.add(l)
		for l in self.dis_net.layers:
			self.combined_net.add(l)
		self.combined_net.to_device(self.dev)

	def train(self):
		train_data, _, _, _, _, _ = load_data(self.dataset_filepath)
		opt_0 = optimizer.Adam(lr=self.learning_rate) # optimizer for discriminator 
		opt_1 = optimizer.Adam(lr=self.learning_rate) # optimizer for generator, aka the combined model
		for (p, specs) in zip(self.dis_net.param_names(), self.dis_net.param_specs()):
			opt_0.register(p, specs)
		for (p, specs) in zip(self.gen_net.param_names(), self.gen_net.param_specs()):
			opt_1.register(p, specs)

		for epoch in range(self.epochs):
			for d_step in range(self.d_steps):
				idx = np.random.randint(0, train_data.shape[0], self.batch_size)
				real_imgs = train_data[idx]
				real_imgs = tensor.from_numpy(real_imgs)
				real_imgs.to_device(self.dev)
				noise = tensor.Tensor((self.batch_size, self.noise_size))
				noise.uniform(-1, 1)
				noise.to_device(self.dev)
				fake_imgs = self.gen_net.forward(flag=False, x=noise)
				substrahend = tensor.Tensor((real_imgs.shape[0], 1))
				substrahend.set_value(1.0)
				substrahend.to_device(self.dev)
				grads, (d_loss_real, _) = self.dis_net.train(real_imgs, substrahend)
				for (s, p ,g) in zip(self.dis_net.param_names(), self.dis_net.param_values(), grads):
					opt_0.apply_with_lr(epoch, self.learning_rate, g, p, str(s), epoch)
				substrahend.set_value(-1.0)
				grads, (d_loss_fake, _) = self.dis_net.train(fake_imgs, substrahend)
				for (s, p ,g) in zip(self.dis_net.param_names(), self.dis_net.param_values(), grads):
					opt_0.apply_with_lr(epoch, self.learning_rate, g, p, str(s), epoch)
				d_loss = d_loss_real + d_loss_fake
			
			for g_step in range(self.g_steps): 
				noise = tensor.Tensor((self.batch_size, self.noise_size))
				noise.uniform(-1, 1)
				noise.to_device(self.dev)
				substrahend = tensor.Tensor((real_imgs.shape[0], 1))
				substrahend.set_value(0.0)
				substrahend.to_device(self.dev)
				grads, (g_loss, _) = self.combined_net.train(noise, substrahend)
				for (s, p ,g) in zip(self.gen_net.param_names(), self.gen_net.param_values(), grads):
					opt_1.apply_with_lr(epoch, self.learning_rate, g, p, str(s), epoch)
			
			if epoch % self.interval == 0:
				self.save_image(epoch)
				print_log('The {} epoch, G_LOSS: {}, D_LOSS: {}'.format(epoch, g_loss, d_loss))

	def save_image(self, epoch):
		rows = 5
		cols = 5
		channels = self.channels
		noise = tensor.Tensor((rows*cols*channels, self.noise_size))
		noise.uniform(-1,1)
		noise.to_device(self.dev)
		gen_imgs = self.gen_net.forward(flag=False, x=noise)
		gen_imgs = tensor.to_numpy(gen_imgs)
		show_imgs = np.reshape(gen_imgs, (gen_imgs.shape[0], self.rows, self.cols, self.channels))
		fig, axs = plt.subplots(rows, cols)
		cnt = 0
		for r in range(rows):
			for c in range(cols):
				axs[r,c].imshow(show_imgs[cnt, :, :, 0], cmap='gray')
				axs[r,c].axis('off')
				cnt += 1
		fig.savefig("{}{}.png".format(self.file_dir, epoch))
		plt.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train GAN over MNIST')
	parser.add_argument('filepath',  type=str, help='the dataset path')
	parser.add_argument('--use_gpu', action='store_true')
	args = parser.parse_args()
	
	if args.use_gpu:
		print('Using GPU')
		dev = device.create_cuda_gpu()
		layer.engine = 'cudnn'
	else:
		print('Using CPU')
		dev = device.get_default_device()
		layer.engine = 'singacpp'

	if not os.path.exists('lsgan_images/'):
		os.makedirs('lsgan_images/')

	rows = 28
	cols = 28
	channels = 1
	noise_size = 100
	hidden_size = 128
	batch = 128
	interval = 1000
	learning_rate = 0.001
	epochs = 1000000
	d_steps = 3
	g_steps = 1
	dataset_filepath = 'mnist.pkl.gz'
	file_dir = 'lsgan_images/'
	lsgan = LSGAN(dev, rows, cols, channels, noise_size, hidden_size, batch, interval, 
		learning_rate, epochs, d_steps, g_steps, dataset_filepath, file_dir)
	lsgan.train()