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

import re
import os
import pickle
import urllib
import tarfile
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
'''
    data collection preprocessing constants
'''
download_dir = '/tmp/'
preprocessed_imdb_data_fp = download_dir + 'imdb_processed.pickle'
imdb_dataset_link = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
google_news_pretrain_embeddings_link = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"


def pad_batch(b, seq_limit):
    ''' convert a batch of encoded sequence
        to pretrained word vectors from the embed weights (lookup dictionary)
    '''
    batch_seq = []
    batch_senti_onehot = []
    batch_senti = []
    for r in b:
        # r[0] encoded sequence
        # r[1] label 1 or 0
        encoded = None
        if len(r[0]) >= seq_limit:
            encoded = r[0][:seq_limit]
        else:
            encoded = r[0] + [0] * (seq_limit - len(r[0]))

        batch_seq.append(encoded)
        batch_senti.append(r[1])
        if r[1] == 1:
            batch_senti_onehot.append([0, 1])
        else:
            batch_senti_onehot.append([1, 0])
    batch_senti = np.array(batch_senti).astype(np.float32)
    batch_senti_onehot = np.array(batch_senti_onehot).astype(np.float32)
    batch_seq = np.array(batch_seq).astype(np.int32)
    return batch_seq, batch_senti_onehot, batch_senti


def pad_batch_2vec(b, seq_limit, embed_weights):
    ''' convert a batch of encoded sequence
        to pretrained word vectors from the embed weights (lookup dictionary)
    '''
    batch_seq = []
    batch_senti_onehot = []
    batch_senti = []
    for r in b:
        # r[0] encoded sequence
        # r[1] label 1 or 0
        encoded = None
        if len(r[0]) >= seq_limit:
            encoded = r[0][:seq_limit]
        else:
            encoded = r[0] + [0] * (seq_limit - len(r[0]))

        batch_seq.append([embed_weights[idx] for idx in encoded])
        batch_senti.append(r[1])
        if r[1] == 1:
            batch_senti_onehot.append([0, 1])
        else:
            batch_senti_onehot.append([1, 0])
    batch_senti = np.array(batch_senti).astype(np.float32)
    batch_senti_onehot = np.array(batch_senti_onehot).astype(np.float32)
    batch_seq = np.array(batch_seq).astype(np.float32)
    return batch_seq, batch_senti_onehot, batch_senti


def check_exist_or_download(url):
    ''' download data into tmp '''
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename


def unzip_data(download_dir, data_gz):
    data_dir = download_dir + 'aclImdb'
    if not os.path.exists(data_dir):
        print("extracting %s to %s" % (download_dir, data_dir))
        with tarfile.open(data_gz) as tar:
            tar.extractall(download_dir)
    return data_dir


def strip_html(text):
    ''' lambda fn for cleaning html '''
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    ''' lambda fn for cleaning square brackets'''
    return re.sub('\[[^]]*\]', '', text)


def remove_special_characters(text, remove_digits=True):
    ''' lambda fn for removing special char '''
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def simple_stemmer(text):
    ''' lambda fn for stemming '''
    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text, tokenizer, stopword_list, is_lower_case=False):
    ''' lambda fn for removing stopwrods '''
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [
            token for token in tokens if token not in stopword_list
        ]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list
        ]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def tokenize(x):
    ''' lambda fn for tokenize sentences '''
    ret = []
    for w in x.split(" "):
        if w != '':
            ret.append(w)
    return ret


def encode_token(words, wv, w2i):
    ''' lambda fn for encoding string seq to int seq 
        args: 
            wv: word vector lookup dictionary
            w2i: word2index lookup dictionary
    '''
    ret = []
    for w in words:
        if w in wv:
            ret.append(w2i[w])
    return ret


def preprocess():
    ''' collect and preprocess raw data from acl Imdb dataset
    '''
    nltk.download('stopwords')

    print("preparing raw imdb data")
    data_gz = check_exist_or_download(imdb_dataset_link)
    data_dir = unzip_data(download_dir, data_gz)

    # imdb dirs
    # vocab_f = data_dir + '/imdb.vocab'
    train_pos_dir = data_dir + '/train/pos/'
    train_neg_dir = data_dir + '/train/neg/'
    test_pos_dir = data_dir + '/test/pos/'
    test_neg_dir = data_dir + '/test/neg/'

    # nltk helpers
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')

    # load pretrained word2vec binary
    print("loading pretrained word2vec")
    google_news_pretrain_fp = check_exist_or_download(
        google_news_pretrain_embeddings_link)
    wv = KeyedVectors.load_word2vec_format(google_news_pretrain_fp, binary=True)

    # parse flat files to memory
    data = []
    for data_dir, label in [(train_pos_dir, 1), (train_neg_dir, 0),
                            (test_pos_dir, 1), (test_neg_dir, 0)]:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename),
                          "r",
                          encoding="utf-8") as fhdl:
                    data.append((fhdl.read(), label))

    # text review cleaning
    print("cleaning text review")
    imdb_data = pd.DataFrame(data, columns=["review", "label"])
    imdb_data['review'] = imdb_data['review'].apply(strip_html)
    imdb_data['review'] = imdb_data['review'].apply(
        remove_between_square_brackets)
    imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)
    imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)
    imdb_data['review'] = imdb_data['review'].apply(remove_stopwords,
                                                    args=(tokenizer,
                                                          stopword_list))
    imdb_data['token'] = imdb_data['review'].apply(tokenize)

    # build  word2index and index2word
    w2i = dict()
    i2w = dict()

    # add vocab <pad> as index 0
    w2i["<pad>"] = 0
    i2w[0] = "<pad>"

    idx = 1  # start from idx 1
    for index, row in imdb_data['token'].iteritems():
        for w in row:
            if w in wv and w not in w2i:
                w2i[w] = idx
                i2w[idx] = w
                assert idx < 28241
                idx += 1
    assert len(w2i) == len(i2w)
    print("vocab size: ", len(w2i))

    # encode tokens to int
    imdb_data['encoded'] = imdb_data['token'].apply(encode_token,
                                                    args=(wv, w2i))

    # select word vector weights for embedding layer from vocab
    embed_weights = []
    for w in w2i.keys():
        val = None
        if w in wv:
            val = wv[w]
        else:
            val = np.zeros([
                300,
            ])
        embed_weights.append(val)
    embed_weights = np.array(embed_weights)
    print("embedding layer lookup weight shape: ", embed_weights.shape)

    # split into train and test
    train_data = imdb_data[['encoded', 'label']].values
    train, val = train_test_split(train_data, test_size=0.33, random_state=42)

    # save preprocessed for training
    imdb_processed = {
        "train": train,
        "val": val,
        "embed_weights": embed_weights,
        "w2i": w2i,
        "i2w": i2w
    }
    print("saving preprocessed file to ", preprocessed_imdb_data_fp)
    with open(preprocessed_imdb_data_fp, 'wb') as handle:
        pickle.dump(imdb_processed, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    preprocess()
