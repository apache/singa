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


class TransformerEncoder(layer.Layer):
    """TransformerEncoder is a stack of N encoder layers
        Args:
           src_n_token: the source vocab size
           d_model: the number of expected features in the encoder inputs (default=512).
           n_head: the number of heads in the multi head attention models (default=8).
           dim_feedforward: the dimension of the feedforward network model (default=2048).
           n_layers: the number of sub-encoder-layers in the encoder (default=6).
    """

    def __init__(self, src_n_token, d_model=512, n_head=8, dim_feedforward=2048, n_layers=6):
        super(TransformerEncoder, self).__init__()
        self.src_n_token = src_n_token
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers

        # input_emb / pos_emb / n-encoder layers
        self.input_emb = layer.Embedding(input_dim=src_n_token, output_dim=d_model)
        self.pos_emb = layer.Embedding(input_dim=src_n_token, output_dim=d_model)
        self.layers = []
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward))

    def forward(self, enc_inputs):
        """Pass the input through the encoder in turn.
        Args:
            enc_inputs: the sequence to the encoder (required).   [batch_size, src_len]
        """
        # [batch_size, src_len, d_model]
        word_emb = self.input_emb(enc_inputs)

        self.pos_emb.initialize(enc_inputs)
        self.pos_emb.from_pretrained(W=TransformerEncoder._get_sinusoid_encoding_table(self.src_n_token, self.d_model), freeze=True)
        # [batch_size, src_len, d_model]
        pos_emb = self.pos_emb(enc_inputs)
        # enc_outputs [batch_size, src_len, d_model]
        enc_outputs = autograd.add(word_emb, pos_emb)

        # enc_self_attn_mask [batch_size, src_len, src_len]
        enc_self_attn_mask = TransformerEncoder._get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

    @staticmethod
    def _get_attn_pad_mask(seq_q, seq_k):
        """
        Args:
            seq_q: [batch_size, seq_len]
            seq_k: [batch_size, seq_len]
        Returns: [batch_size, seq_len, seq_len]
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
    def _get_sinusoid_encoding_table(n_position, d_model):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)], np.float32)
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return tensor.Tensor(data=sinusoid_table, requires_grad=False)


class TransformerEncoderLayer(layer.Layer):
    def __init__(self, d_model=512, n_head=8, dim_feedforward=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.enc_self_attn = MultiHeadAttention(d_model, n_head)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, dim_feedforward=dim_feedforward, bias=False)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Args:
            enc_inputs: [batch_size, src_len, d_model]
            enc_self_attn_mask: [batch_size, src_len, src_len]

        Returns:
            enc_outputs: [batch_size, src_len, d_model]
            attn: [batch_size, n_heads, src_len, src_len]
        """
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


def matmul4d(x1, x2):
    batchs, heads = x1.shape[0], x1.shape[1]
    ys = []
    for b in range(batchs):
        x1b, x2b = autograd.squeeze(x1[b]), autograd.squeeze(x2[b])
        yb = []
        for h in range(heads):
            x1h, x2h = autograd.squeeze(x1b[h]), autograd.squeeze(x2b[h])
            yh = autograd.matmul(x1h, x2h)
            yh = autograd.unsqueeze(yh, axis=[0])
            yb.append(yh)
        yb = autograd.cat(yb, axis=0)
        yb = autograd.unsqueeze(yb, axis=[0])
        ys.append(yb)
    y = autograd.cat(ys, axis=0)
    return y


class MultiHeadAttention(layer.Layer):
    def __init__(self, d_model=512, n_head=8):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_head
        assert (
                self.d_k * n_head == d_model
        ), "embed_dim must be divisible by num_heads"
        self.d_model = d_model
        self.d_v = self.d_k
        self.n_head = n_head
        self.W_Q = Linear3D(d_model, self.d_k * n_head)
        self.W_K = Linear3D(d_model, self.d_k * n_head)
        self.W_V = Linear3D(d_model, self.d_v * n_head)

        self.scaled_dot_product_attention = ScaledDotProductAttention(d_model, n_head)
        self.linear = Linear3D(self.d_v * n_head, d_model)
        self.add = layer.Add()
        self.layer_norm = LayerNorm(d_model)

    def forward(self, query, key, value, attn_mask):
        """
        Args:
            query: [batch_size, len_q, d_model]
            key: [batch_size, len_k, d_model]
            value: [batch_size, len_v(=len_k), d_model]
            attn_mask: [batch_size, seq_len, seq_len]
        Returns:
        """
        residual = query
        batch_size = query.shape[0]

        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(query)
        Q = autograd.reshape(Q, [batch_size, -1, self.n_head, self.d_k])
        Q = autograd.transpose(Q, [0, 2, 1, 3])

        K = self.W_K(key)
        K = autograd.reshape(K, [batch_size, -1, self.n_head, self.d_k])
        K = autograd.transpose(K, [0, 2, 1, 3])

        V = self.W_V(value)
        V = autograd.reshape(V, [batch_size, -1, self.n_head, self.d_v])
        V = autograd.transpose(V, [0, 2, 1, 3])

        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = MultiHeadAttention._get_attn_mask(attn_mask, self.n_head)

        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, seq_len, seq_len]
        context, attn = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        context = autograd.transpose(context, [0, 2, 1, 3])
        # context: [batch_size, len_q, n_heads * d_v]
        context = autograd.reshape(context, [batch_size, -1, self.n_head * self.d_v])

        output = self.linear(context)
        output = self.add(output, residual)
        # [batch_size, len_q, d_model]
        output = self.layer_norm(output)
        return output, attn

    @staticmethod
    def _get_attn_mask(attn_mask, n_head):
        batch_size, seq_q_len,seq_k_len = attn_mask.shape[0], attn_mask.shape[1], attn_mask.shape[2]
        attn_mask_np = tensor.to_numpy(attn_mask)
        attn_mask_np = np.expand_dims(attn_mask_np, axis=1)
        attn_mask_np = np.broadcast_to(attn_mask_np, (batch_size, n_head, seq_q_len, seq_k_len))
        return tensor.from_numpy(attn_mask_np)


class ScaledDotProductAttention(layer.Layer):
    def __init__(self, d_model=512, n_head=8):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_model // n_head
        assert (
                self.d_k * n_head == d_model
        ), "embed_dim must be divisible by num_heads"

    def forward(self, query, key, value, attn_mask):
        """
        Args:
            query: [batch_size, n_heads, len_q, d_k]
            key: [batch_size, n_heads, len_k, d_k]
            value: [batch_size, n_heads, len_v(=len_k), d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]
        Returns:
        """

        K_trans = autograd.transpose(key, [0, 1, 3, 2])

        # scores : [batch_size, n_heads, len_q, len_k]
        # query [batch_size, n_heads, len_q, d_k]
        # k^T   [batch_size, n_heads, d_k, len_k]
        scores = matmul4d(query, K_trans)
        d_k_sqrt = Tensor(shape=(1,), requires_grad=False, stores_grad=False)
        d_k_sqrt.set_value(np.sqrt(self.d_k))
        scores = autograd.div(scores, d_k_sqrt)

        mask_fill = Tensor(shape=attn_mask.shape, data=np.full(attn_mask.shape, -1e6, dtype=np.float32), requires_grad=False, stores_grad=False)
        attn_mask_np = tensor.to_numpy(attn_mask)
        scores = autograd.where(mask_fill, scores, attn_mask_np)

        attn = autograd.softmax(scores, axis=-1)
        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]  value: [batch_size, n_heads, len_v(=len_k), d_v]
        context = matmul4d(attn, value)
        return context, attn


class PoswiseFeedForwardNet(layer.Layer):
    def __init__(self, d_model=512, dim_feedforward=2048, bias=False):
        super(PoswiseFeedForwardNet, self).__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.bias = bias

        self.linear1 = Linear3D(d_model, dim_feedforward, bias=bias)
        self.relu = layer.ReLU()
        self.linear2 = Linear3D(dim_feedforward, d_model, bias=bias)
        self.add = layer.Add()
        self.norm = LayerNorm(d_model)

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.linear1(inputs)
        output = self.relu(output)
        output = self.linear2(output)
        # [batch_size, seq_len, d_model]
        output = self.add(output, residual)
        output = self.norm(output)
        return output


class LayerNorm(layer.Layer):
    def __init__(self, n_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.n_features = n_features
        self.eps = eps

    def initialize(self, x):
        shape = (self.n_features,)
        self.Gamma = Tensor(shape=shape, dtype=x.dtype, requires_grad=False, stores_grad=False)
        self.Beta = Tensor(shape=shape, dtype=x.dtype, requires_grad=False, stores_grad=False)
        self.Gamma.set_value(1.0)
        self.Beta.set_value(0.0)

    def forward(self, x):
        # x: input tensor with shape [batch_size, n_features]
        # x_normalized = (x - tensor.from_numpy(self.mean)) / tensor.from_numpy(np.sqrt(self.var + self.eps))
        # y = self.gamma * x_normalized + self.beta
        mean = np.mean(tensor.to_numpy(x), axis=-1, keepdims=True)
        var = np.var(tensor.to_numpy(x), axis=-1, keepdims=True)

        sub1 = tensor.from_numpy(mean)
        div1 = tensor.from_numpy(np.sqrt(var + self.eps))
        x_normalized = autograd.div(autograd.sub(x, sub1), div1)
        y = autograd.mul(self.Gamma, x_normalized)
        y = autograd.add(y, self.Beta)
        return y


class Linear3D(layer.Layer):
    """
    Generate a Linear3D operator
    """

    # TODO: replace current with
    #   def __init__(self, out_features, bias=True):
    def __init__(self, out_features, *args, bias=False, **kwargs):
        """
        Args:
            ut_channels: int, the channel of output, also is the number of
                filters
            bias: bool
        """
        super(Linear3D, self).__init__()
        self.out_features = out_features

        # TODO: for backward compatibility, to remove
        if len(args) > 0:
            self.in_features = out_features
            self.out_features = args[0]
        if len(args) > 1:
            self.bias = args[1]
        else:
            self.bias = bias

    def initialize(self, x):
        self.in_features = x.shape[-1]
        w_shape = (self.in_features, self.out_features)
        b_shape = (self.out_features,)

        self.W = Tensor(shape=w_shape,
                        dtype=x.dtype,
                        requires_grad=True,
                        stores_grad=True)
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        self.W.gaussian(0.0, std)

        if self.bias:
            self.b = Tensor(shape=b_shape,
                            dtype=x.dtype,
                            requires_grad=True,
                            stores_grad=True)
            self.b.set_value(0.0)
        else:
            self.b = None

    def forward(self, x):
        if self.b:
            self.device_check(x, self.W, self.b)
            self.dtype_check(x, self.W, self.b)
        else:
            self.device_check(x, self.W)
            self.dtype_check(x, self.W)

        assert x.shape[-1] == self.W.shape[0], (
                "Linear3D layer expects input features size %d received %d" %
                (self.W.shape[0], x.shape[-1]))

        ys = []
        batch = x.shape[0]
        for i in range(batch):
            xi = autograd.squeeze(x[i])
            yi = autograd.matmul(xi, self.W)
            if self.bias:
                yi = autograd.add_bias(yi, self.b, axis=0)
            yi = autograd.unsqueeze(yi, axis=[0])
            ys.append(yi)
        y = autograd.cat(ys, axis=0)
        return y

    def get_params(self):
        if self.bias:
            return {self.W.name: self.W, self.b.name: self.b}
        else:
            return {self.W.name: self.W}

    def set_params(self, parameters):
        self.W.copy_from(parameters[self.W.name])
        if self.bias:
            self.b.copy_from(parameters[self.b.name])
