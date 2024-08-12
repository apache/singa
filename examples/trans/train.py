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

import random
import time
import argparse
import numpy as np
from singa import device
from singa import tensor
from singa import opt
from model import Transformer
import matplotlib.pyplot as plt
from data import CmnDataset


def run(args):
    dev = device.create_cpu_device()
    dev.SetRandSeed(args.seed)
    np.random.seed(args.seed)

    batch_size = args.batch_size
    cmn_dataset = CmnDataset(path="cmn-eng/"+args.dataset, shuffle=args.shuffle, batch_size=batch_size, train_ratio=0.8)

    print("【step-0】 prepare dataset...")
    src_vocab_size, tgt_vocab_size = cmn_dataset.en_vab_size, cmn_dataset.cn_vab_size
    src_len, tgt_len = cmn_dataset.src_max_len+1, cmn_dataset.tgt_max_len+1
    pad = cmn_dataset.cn_vab["<pad>"]
    # train set
    train_size = cmn_dataset.train_size
    train_max_batch = train_size // batch_size
    if train_size % batch_size > 0:
        train_max_batch += 1

    # test set
    test_size = cmn_dataset.test_size
    test_max_batch = test_size // batch_size
    if test_size % batch_size > 0:
        test_max_batch += 1
    print("【step-0】 src_vocab_size: %d, tgt_vocab_size: %d, src_max_len: %d, tgt_max_len: %d, "
          "train_size: %d, test_size: %d, train_max_batch: %d, test_max_batch: %d" %
          (src_vocab_size, tgt_vocab_size, src_len, tgt_len, train_size, test_size, train_max_batch, test_max_batch))

    print("【step-1】 prepare transformer model...")
    model = Transformer(src_n_token=src_vocab_size,
                        tgt_n_token=tgt_vocab_size,
                        d_model=args.d_model,
                        n_head=args.n_head,
                        dim_feedforward=args.dim_feedforward,
                        n_layers=args.n_layers)

    optimizer = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5)
    model.set_optimizer(optimizer)
    print("【step-1】 src_n_token: %d, tgt_n_token: %d, d_model: %d, n_head: %d, dim_feedforward: %d, n_layers: %d, lr: %f"
          % (src_vocab_size, tgt_vocab_size, args.d_model, args.n_head, args.dim_feedforward, args.n_layers, args.lr))

    tx_enc_inputs = tensor.Tensor((batch_size, src_len), dev, tensor.int32,
                                  np.zeros((batch_size, src_len), dtype=np.int32))
    tx_dec_inputs = tensor.Tensor((batch_size, tgt_len), dev, tensor.int32,
                                  np.zeros((batch_size, tgt_len), dtype=np.int32))
    ty_dec_outputs = tensor.Tensor((batch_size, tgt_len), dev, tensor.int32,
                                   np.zeros((batch_size, tgt_len), dtype=np.int32))
    # model.compile([tx_enc_inputs, tx_dec_inputs], is_train=True)

    print("【step-2】 training start...")
    train_epoch_avg_loss_history = []
    train_epoch_avg_acc_history = []
    test_epoch_avg_acc_history = []
    for epoch in range(args.max_epoch):
        # ok = input("Train[Yes/No]")
        # if ok == "No":
        #     break
        model.train()
        model.graph(mode=False, sequential=False)
        train_epoch_total_loss = 0
        train_epoch_total_acc = 0.0
        print("【Train epoch %d】 Start..." % epoch)
        start_time = time.time()
        for bat in range(train_max_batch):
            x_enc_inputs, x_dec_inputs, y_dec_outputs = cmn_dataset.get_batch_data(batch=bat, mode='train')
            tx_enc_inputs.copy_from_numpy(x_enc_inputs)
            tx_dec_inputs.copy_from_numpy(x_dec_inputs)
            ty_dec_outputs.copy_from_numpy(y_dec_outputs)
            out, loss, acc = model(tx_enc_inputs, tx_dec_inputs, ty_dec_outputs, pad)
            loss_np = tensor.to_numpy(loss)
            batch_loss = loss_np[0]
            train_epoch_total_loss += batch_loss
            train_epoch_total_acc += acc
            if bat % 5 == 0:
                end_time = time.time()
                print("[Train epoch-%d] [%d/%d] batch loss: [%.6f], acc: [%.3f %%] time:[%.6fs]" %
                      (epoch, bat, train_max_batch, batch_loss, acc*100.0, end_time-start_time))
                start_time = time.time()
        train_epoch_avg_loss = train_epoch_total_loss / train_max_batch
        train_epoch_avg_acc = train_epoch_total_acc / train_max_batch
        train_epoch_avg_loss_history.append(train_epoch_avg_loss)
        train_epoch_avg_acc_history.append(train_epoch_avg_acc)
        print("[Train epoch-%d] avg loss: [%.6f], avg acc: [%.3f %%]" % (epoch, train_epoch_avg_loss, train_epoch_avg_acc*100.0))
        print("【Train Epoch %d】 End" % (epoch))
        # eval
        model.eval()
        print("【Test Eval】 Start...")
        avg_acc = 0.0
        for bat in range(test_max_batch):
            x_enc_inputs, x_dec_inputs, y_dec_outputs = cmn_dataset.get_batch_data(batch=bat, mode='test')
            tx_enc_inputs.copy_from_numpy(x_enc_inputs)
            tx_dec_inputs.copy_from_numpy(x_dec_inputs)
            # ty_dec_outputs.copy_from_numpy(y_dec_outputs)
            # y_dec_outputs [batch_size, tgt_len]
            # out [batch_size, tgt_len, tgt_vocab_size]
            out, _, _, _ = model(tx_enc_inputs, tx_dec_inputs)
            out_np = tensor.to_numpy(out)
            out_np = np.reshape(out_np, (-1, out_np.shape[-1]))
            pred_np = np.argmax(out_np, -1)
            y_dec_outputs = np.reshape(y_dec_outputs, -1)
            y_label_mask = y_dec_outputs != pad
            correct = pred_np == y_dec_outputs
            acc = np.sum(y_label_mask * correct) / np.sum(y_label_mask)
            avg_acc += acc
        avg_acc = avg_acc / test_max_batch
        test_epoch_avg_acc_history.append(avg_acc)
        print("[Test epoch-%d] avg acc: %.3f %%" % (epoch, avg_acc*100.0))
    print("【Test Eval】 End...")
    plt.subplot(2, 1, 1)
    plt.plot(train_epoch_avg_loss_history, 'r-', label="train loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_epoch_avg_acc_history, 'r-', label="train acc")
    plt.plot(test_epoch_avg_acc_history, 'b-', label="test acc")
    plt.legend()
    plt.show()
    timestamp = time.time()
    plt.savefig("batch_train_loss" + str(timestamp).replace(".", "_") + ".png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Transformer Model.")
    parser.add_argument('--dataset', choices=['cmn.txt', 'cmn-15000.txt',
                                              'cmn-2000.txt'],  default='cmn-2000.txt')
    parser.add_argument('--max-epoch', default=100, type=int, help='maximum epochs.', dest='max_epoch')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size', dest='batch_size')
    parser.add_argument('--shuffle', default=True, type=bool, help='shuffle the dataset', dest='shuffle')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate', dest='lr')
    parser.add_argument('--seed', default=0, type=int, help='random seed', dest='seed')
    parser.add_argument('--d_model', default=512, type=int, help='transformer model d_model', dest='d_model')
    parser.add_argument('--n_head', default=8, type=int, help='transformer model n_head', dest='n_head')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='transformer model dim_feedforward', dest='dim_feedforward')
    parser.add_argument('--n_layers', default=6, type=int, help='transformer model n_layers', dest='n_layers')
    args = parser.parse_args()
    print(args)
    run(args)
