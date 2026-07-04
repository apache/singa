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

import re
import numpy as np
from collections import Counter


class Vocab:
    """
    The class of vocab, include 2 dicts of token to index and index to token
    """
    def __init__(self, sentences):
        """
        Args:
            sentences: a 2-dim list
        """
        flatten = lambda lst: [item for sublist in lst for item in sublist]
        self.sentence = sentences
        self.token2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
        self.token2index.update({
            token: index + 4
            for index, (token, freq) in enumerate(
                sorted(Counter(flatten(self.sentence)).items(), key=lambda x: x[1], reverse=True))
        })
        self.index2token = {index: token for token, index in self.token2index.items()}

    def __getitem__(self, query):
        if isinstance(query, str):
            return self.token2index.get(query, self.token2index.get('<unk>'))
        elif isinstance(query, (int, np.int32, np.int64)):
            return self.index2token.get(query, '<unk>')
        elif isinstance(query, (list, tuple, np.ndarray)):
            return [self.__getitem__(item) for item in query]
        else:
            raise ValueError("The type of query is invalid.")

    def __len__(self):
        return len(self.index2token)


class CmnDataset:
    def __init__(self, path, shuffle=False, batch_size=32, train_ratio=0.8, random_seed=0):
        """
        cmn dataset, download from https://www.manythings.org/anki/, contains 29909 Chinese and English translation
        pairs, the pair format: English + TAB + Chinese + TAB + Attribution
        Args:
            path: the path of the dataset
            shuffle: shuffle the dataset, default False
            batch_size: the size of every batch, default 32
            train_ratio: the proportion of the training set to the total data set, default 0.8
            random_seed: the random seed, used for shuffle operation, default 0
        """
        src_max_len, tgt_max_len, src_sts, tgt_sts = CmnDataset._split_sentences(path)
        en_vab, cn_vab = Vocab(src_sts), Vocab(tgt_sts)
        src_np, tgt_in_np, tgt_out_np = CmnDataset._encoding_stc(src_sts, tgt_sts, src_max_len, tgt_max_len,
                                                                 en_vab, cn_vab)

        self.src_max_len, self.tgt_max_len = src_max_len, tgt_max_len
        self.en_vab, self.cn_vab = en_vab, cn_vab
        self.en_vab_size, self.cn_vab_size = len(en_vab), len(cn_vab)

        self.src_inputs, self.tgt_inputs, self.tgt_outputs = src_np, tgt_in_np, tgt_out_np

        self.shuffle, self.random_seed = shuffle, random_seed

        assert batch_size > 0, "The number of batch_size must be greater than 0"
        self.batch_size = batch_size

        assert (0 < train_ratio <= 1.0), "The number of train_ratio must be in (0.0, 1.0]"
        self.train_ratio = train_ratio

        self.total_size = len(src_np)
        self.train_size = int(self.total_size * train_ratio)
        self.test_size = self.total_size - self.train_size

        if shuffle:
            index = [i for i in range(self.total_size)]
            np.random.seed(self.random_seed)
            np.random.shuffle(index)

            self.src_inputs = src_np[index]
            self.tgt_inputs = tgt_in_np[index]
            self.tgt_outputs = tgt_out_np[index]

        self.train_src_inputs, self.test_src_inputs = self.src_inputs[:self.train_size], self.src_inputs[self.train_size:]
        self.train_tgt_inputs, self.test_tgt_inputs = self.tgt_inputs[:self.train_size], self.tgt_inputs[self.train_size:]
        self.train_tgt_outputs, self.test_tgt_outputs = self.tgt_outputs[:self.train_size], self.tgt_outputs[self.train_size:]

    @staticmethod
    def _split_sentences(path):
        en_max_len, cn_max_len = 0, 0
        en_sts, cn_sts = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line_split = line.split('\t')
                line_split[0] = re.sub(r'[^\w\s\'-]', '', line_split[0])
                line_split[0] = line_split[0].lower()
                # [\u4e00-\u9fa5] matching Chinese characters
                line_split[1] = re.sub("[^\u4e00-\u9fa5]", "", line_split[1])

                en_stc = line_split[0].split(' ')
                cn_stc = [word for word in line_split[1]]
                en_sts.append(en_stc)
                cn_sts.append(cn_stc)
                en_max_len = max(en_max_len, len(en_stc))
                cn_max_len = max(cn_max_len, len(cn_stc))
        return en_max_len, cn_max_len, en_sts, cn_sts

    @staticmethod
    def _encoding_stc(src_tokens, tgt_tokens, src_max_len, tgt_max_len, src_vocab, tgt_vocab):
        src_list = []
        for line in src_tokens:
            if len(line) > src_max_len:
                line = line[:src_max_len]
            lst = src_vocab[line + ['<pad>'] * (src_max_len + 1 - len(line))]
            src_list.append(lst)
        tgt_in_list, tgt_out_list = [], []
        for line in tgt_tokens:
            if len(line) > tgt_max_len:
                line = line[:tgt_max_len]
            in_lst = tgt_vocab[['<bos>'] + line + ['<pad>'] * (tgt_max_len - len(line))]
            out_lst = tgt_vocab[line + ['<eos>'] + ['<pad>'] * (tgt_max_len - len(line))]
            tgt_in_list.append(in_lst)
            tgt_out_list.append(out_lst)
        src_np = np.asarray(src_list, dtype=np.int32)
        tgt_in_np = np.asarray(tgt_in_list, dtype=np.int32)
        tgt_out_np = np.asarray(tgt_out_list, dtype=np.int32)
        return src_np, tgt_in_np, tgt_out_np

    def get_batch_data(self, batch, mode='train'):
        assert (mode == 'train' or mode == 'test'), "The mode must be 'train' or 'test'."
        total_size = self.train_size
        if mode == 'test':
            total_size = self.test_size

        max_batch = total_size // self.batch_size
        if total_size % self.batch_size > 0:
            max_batch += 1
        assert batch < max_batch, "The batch number is out of bounds."

        low = batch * self.batch_size
        if (batch + 1) * self.batch_size < total_size:
            high = (batch + 1) * self.batch_size
        else:
            high = total_size
        if mode == 'train':
            if high-low != self.batch_size:
                return (np.concatenate((self.train_src_inputs[low:high], self.train_src_inputs[:self.batch_size-high+low]), axis=0),
                        np.concatenate((self.train_tgt_inputs[low:high], self.train_tgt_inputs[:self.batch_size-high+low]), axis=0),
                        np.concatenate((self.train_tgt_outputs[low:high], self.train_tgt_outputs[:self.batch_size-high+low]), axis=0))
            else:
                return self.train_src_inputs[low:high], self.train_tgt_inputs[low:high], self.train_tgt_outputs[low:high]
        else:
            if high-low != self.batch_size:
                return (np.concatenate((self.test_src_inputs[low:high], self.test_src_inputs[:self.batch_size-high+low]), axis=0),
                        np.concatenate((self.test_tgt_inputs[low:high], self.test_tgt_inputs[:self.batch_size-high+low]), axis=0),
                        np.concatenate((self.test_tgt_outputs[low:high], self.test_tgt_outputs[:self.batch_size-high+low]), axis=0))
            return self.test_src_inputs[low:high], self.test_tgt_inputs[low:high], self.test_tgt_outputs[low:high]

    def __len__(self):
        return self.src_inputs.shape[0]

    def __getitem__(self, idx):
        return self.src_inputs[idx], self.tgt_inputs[idx], self.tgt_outputs[idx]
