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

import math
import numpy as np
from singa import tensor
from singa import autograd
from singa import layer
from singa import model
from singa.tensor import Tensor


class Transformer(model.Model):
    def __init__(self, src_n_token, tgt_n_token, d_model=512, n_head=8, dim_feedforward=2048, n_layers=6):
        """
        Transformer model
        Args:
            src_n_token: the size of source vocab
            tgt_n_token: the size of target vocab
            d_model: the number of expected features in the encoder/decoder inputs (default=512)
            n_head: the number of heads in the multi head attention models (default=8)
            dim_feedforward: the dimension of the feedforward network model (default=2048)
            n_layers: the number of sub-en(de)coder-layers in the en(de)coder (default=6)
        """
        super(Transformer, self).__init__()

        self.opt = None
        self.src_n_token = src_n_token
        self.tgt_n_token = tgt_n_token
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers

        # encoder / decoder / linear
        self.encoder = TransformerEncoder(src_n_token=src_n_token, d_model=d_model, n_head=n_head,
                                          dim_feedforward=dim_feedforward, n_layers=n_layers)
        self.decoder = TransformerDecoder(tgt_n_token=tgt_n_token, d_model=d_model, n_head=n_head,
                                          dim_feedforward=dim_feedforward, n_layers=n_layers)

        self.linear3d = Linear3D(in_features=d_model, out_features=tgt_n_token, bias=False)

        self.soft_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, enc_inputs, dec_inputs):
        """
        Args:
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]

        """
        # enc_outputs: [batch_size, src_len, d_model],
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.linear3d(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

    def train_one_batch(self, enc_inputs, dec_inputs, dec_outputs, pad):
        out, _, _, _ = self.forward(enc_inputs, dec_inputs)
        shape = out.shape[-1]
        out = autograd.reshape(out, [-1, shape])

        out_np = tensor.to_numpy(out)
        preds_np = np.argmax(out_np, -1)

        dec_outputs_np = tensor.to_numpy(dec_outputs)
        dec_outputs_np = dec_outputs_np.reshape(-1)

        y_label_mask = dec_outputs_np != pad
        correct = preds_np == dec_outputs_np
        acc = np.sum(y_label_mask * correct) / np.sum(y_label_mask)
        dec_outputs = tensor.from_numpy(dec_outputs_np)

        loss = self.soft_cross_entropy(out, dec_outputs)
        self.opt(loss)
        return out, loss, acc

    def set_optimizer(self, opt):
        self.opt = opt

class TransformerDecoderLayer(layer.Layer):
    def __init__(self, d_model=512, n_head=8, dim_feedforward=2048):
        super(TransformerDecoderLayer, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward

        self.dec_self_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dec_enc_attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, dim_feedforward=dim_feedforward)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        Args:
            dec_inputs: [batch_size, tgt_len, d_model]
            enc_outputs: [batch_size, src_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class TransformerDecoder(layer.Layer):
    """TransformerDecoder is a stack of N decoder layers
        Args:
            tgt_n_token: the size of target vocab
            d_model: the number of expected features in the decoder inputs (default=512).
            n_head: the number of heads in the multi head attention models (default=8).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            n_layers: the number of sub-decoder-layers in the decoder (default=6).
    """

    def __init__(self, tgt_n_token, d_model=512, n_head=8, dim_feedforward=2048, n_layers=6):
        super(TransformerDecoder, self).__init__()
        self.tgt_n_token = tgt_n_token
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers

        # target_emb / pos_emb / n-layers
        self.target_emb = layer.Embedding(input_dim=tgt_n_token, output_dim=d_model)
        self.target_pos_emb = layer.Embedding(input_dim=tgt_n_token, output_dim=d_model)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(TransformerDecoderLayer(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward))

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        Args:
            dec_inputs: [batch_size, tgt_len]
            enc_inputs: [batch_size, src_len]
            enc_outputs: [batch_size, src_len, d_model]

        """

        # [batch_size, tgt_len, d_model]
        tgt_word_emb = self.target_emb(dec_inputs)
        self.target_pos_emb.initialize(dec_inputs)
        self.target_pos_emb.from_pretrained(W=TransformerDecoder._get_sinusoid_encoding_table(self.tgt_n_token, self.d_model),
                                            freeze=True)
        # [batch_size, tgt_len, d_model]
        tgt_pos_emb = self.target_pos_emb(dec_inputs)
        # [batch_size, tgt_len, d_model]
        dec_outputs = autograd.add(tgt_word_emb, tgt_pos_emb)

        # dec_self_attn_pad_mask  [batch_size, tgt_len, tgt_len]
        dec_self_attn_pad_mask = TransformerDecoder._get_attn_pad_mask(dec_inputs, dec_inputs)
        # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = TransformerDecoder._get_attn_subsequence_mask(dec_inputs)

        # dec_self_attn_mask [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = tensor.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # dec_enc_attn_mask [batch_size, tgt_len, src_len]
        dec_enc_attn_mask = TransformerDecoder._get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len,src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

    @staticmethod
    def _get_attn_pad_mask(seq_q, seq_k):
        """
        Args:
            seq_q: [batch_size, seq_len]
            seq_k: [batch_size, seq_len]
        Returns:
            [batch_size, seq_len, seq_len]
        """

        batch_size, len_q = seq_q.shape
        batch_size, len_k = seq_k.shape
        seq_k_np = tensor.to_numpy(seq_k)
        pad_attn_mask_np = np.where(seq_k_np == 0, 1, 0)
        pad_attn_mask_np.astype(np.int32)
        pad_attn_mask_np = np.expand_dims(pad_attn_mask_np, axis=1)
        pad_attn_mask_np = np.broadcast_to(pad_attn_mask_np, (batch_size, len_q, len_k))
        pad_attn_mask_np = tensor.from_numpy(pad_attn_mask_np)
        return pad_attn_mask_np

    @staticmethod
    def _get_attn_subsequence_mask(seq):
        """
        Args:
            seq: [batch_size, tgt_len]

        Returns:
        """
        attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]

        # generate the upper triangular matrix, [batch_size, tgt_len, tgt_len]
        subsequence_mask = np.triu(np.ones(attn_shape), k=1)
        subsequence_mask.astype(np.int32)
        subsequence_mask = tensor.from_numpy(subsequence_mask)
        return subsequence_mask

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_model):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)], np.float32)
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # Even bits use sine functions
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # Cosine function for odd digits
        return tensor.Tensor(data=sinusoid_table, requires_grad=False)

