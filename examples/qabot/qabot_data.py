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

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random
from deprecated import deprecated

@deprecated(reason="deprecated method: load_vocabulary called")
def load_vocabulary(vocab_path, label_path):
    id_to_word = {}
    with open(vocab_path, 'rb') as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().decode("utf-8").split("\t")
            if d[0] not in id_to_word:
                id_to_word[d[0]] = d[1]

    label_to_ans = {}
    label_to_ans_text = {}
    with open(label_path) as f:
        lines = f.readlines()
        for l in lines:
            label, answer = l.rstrip().split('\t')
            if label not in label_to_ans:
                label_to_ans[label] = answer
                label_to_ans_text[label] = [
                    id_to_word[t] for t in answer.split(' ')
                ]
    return id_to_word, label_to_ans, label_to_ans_text


@deprecated(reason="deprecated method: parse_file called")
def parse_file(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')]  # question
            poss = [label_to_ans_text[t] for t in d[2].split(' ')
                   ]  # ground-truth
            negs = [
                label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]
            ]  # candidate-pool without ground-truth
            for pos in poss:
                data.append((q, pos, negs))
    return data


@deprecated(reason="deprecated method: parse_test_file called")
def parse_test_file(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines[12:]:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')]  # question
            poss = [t for t in d[2].split(' ')]  # ground-truth
            cands = [t for t in d[3].split(' ')]  # candidate-pool
            data.append((q, poss, cands))
    return data

@deprecated(reason="deprecated method: words_text_to_fixed_seqlen_vec called")
def words_text_to_fixed_seqlen_vec(word2vec, words, sentence_length=10):
    sentence_vec = []
    for word in words:
        if len(sentence_vec) >= sentence_length:
            break
        if word in word2vec:
            sentence_vec.append(word2vec[word])
        else:
            sentence_vec.append(np.zeros((300,)))
    while len(sentence_vec) < sentence_length:
        sentence_vec.append(np.zeros((300,)))
    return np.array(sentence_vec, dtype=np.float32)


@deprecated(reason="deprecated method: generate_qa_triplets called")
def generate_qa_triplets(data, num_negs=10):
    tuples = []
    for (q, a_pos, a_negs) in data:
        for i in range(num_negs):
            tpl = (q, a_pos, random.choice(a_negs))
            tuples.append(tpl)
    return tuples

@deprecated(reason="deprecated method: qa_tuples_to_naive_training_format called")
def qa_tuples_to_naive_training_format(wv, tuples):
    training = []
    q_len_limit = 10
    a_len_limit = 100
    for tpl in tuples:
        q, a_pos, a_neg = tpl
        q_vec = words_text_to_fixed_seqlen_vec(wv, q, q_len_limit)
        training.append(
            (q_vec, words_text_to_fixed_seqlen_vec(wv, a_pos, a_len_limit), 1))
        training.append(
            (q_vec, words_text_to_fixed_seqlen_vec(wv, a_neg, a_len_limit), 0))
    return training

@deprecated(reason="deprecated method: triplet_text_to_vec called")
def triplet_text_to_vec(triplet, wv, q_max_len, a_max_len):
    return [
        words_text_to_fixed_seqlen_vec(wv, triplet[0], q_max_len),
        words_text_to_fixed_seqlen_vec(wv, triplet[1], a_max_len),
        words_text_to_fixed_seqlen_vec(wv, triplet[2], a_max_len)
    ]

@deprecated(reason="deprecated method: train_data_gen_fn called")
def train_data_gen_fn(train_triplet_vecs, bs=32):
    q = []
    ap = []
    an = []
    for t in train_triplet_vecs:
        q.append(t[0])
        ap.append(t[1])
        an.append(t[2])
        if len(q) >= bs:
            q = np.array(q)
            ap = np.array(ap)
            an = np.array(an)
            a = np.concatenate([ap, an])
            assert 2 * q.shape[0] == a.shape[0]
            yield q, a
            q = []
            ap = []
            an = []
    # return the rest
    # return np.array(q), np.concatenate([np.array(ap), np.array(an)])


download_dir="/tmp/"
import os
import urllib
def check_exist_or_download(url):
    ''' download data into tmp '''
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename

def unzip_data(download_dir, data_zip):
    data_dir = download_dir + "insuranceQA-master/V2/"
    if not os.path.exists(data_dir):
        print("extracting %s to %s" % (download_dir, data_dir))
        from zipfile import ZipFile
        with ZipFile(data_zip, 'r') as zipObj:
            zipObj.extractall(download_dir)
    return data_dir

def get_label2answer(data_dir):
    import gzip
    label2answer=dict()
    with gzip.open(data_dir+"/InsuranceQA.label2answer.token.encoded.gz") as fin:
        for line in fin:
            pair = line.decode().strip().split("\t")
            idxs = pair[1].split(" ")
            idxs = [int(idx.replace("idx_","")) for idx in idxs]
            label2answer[int(pair[0])] = idxs
    return label2answer

pad_idx = 0
pad_string = "<pad>"
pad_embed = np.zeros((300,))

insuranceqa_train_filename = "/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded.gz"
insuranceqa_test_filename = "/InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded.gz"
insuranceQA_url = "https://github.com/shuzi/insuranceQA/archive/master.zip"
insuranceQA_cache_fp = download_dir+"insuranceQA_cache.pickle"
google_news_pretrain_embeddings_link = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

def get_idx2word(data_dir):
    idx2word = dict()
    with open(data_dir+"vocabulary") as vc_f:
        for line in vc_f:
            pair=line.strip().split("\t")
            idx = int(pair[0].replace("idx_",""))
            idx2word[idx] = pair[1]

    # add padding string to idx2word lookup
    idx2word[pad_idx] = pad_string

    return idx2word

def get_train_raw(data_dir):
    ''' deserialize training data file
        args:
            data_dir: dir of data file
        return:
            train_raw: list of QnA pair, length of list  == number of samples,
                each pair has 3 fields:
                    0 is question sentence idx encoded, use idx2word to decode,
                        idx2vec to get embedding.
                    1 is ans labels, each label corresponds to a ans sentence,
                        use label2answer to decode.
                    2 is top K candidate ans, these are negative ans for
                        training.
    '''
    train_raw = []
    import gzip
    with gzip.open(data_dir+insuranceqa_train_filename) as fin:
        for line in fin:
            tpl = line.decode().strip().split("\t")
            question = [int(idx.replace("idx_","")) for idx in tpl[1].split(" ")]
            ans = [int(label) for label in tpl[2].split(" ")]
            candis = [int(label) for label in tpl[3].split(" ")]
            train_raw.append((question,ans,candis))
    return train_raw

def get_filtered_crop_data(train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec):
    ''' prepare train data to embedded word vector sequence given sequence limit
        return:
            questions_encoded: np ndarray, shape
                (number samples, seq length, vector size)
            poss_encoded: same layout, sequence for positive answer
            negs_encoded: same layout, sequence for negative answer
    '''
    questions = [question for question, answers, candis in train_raw]
    # choose 1 answer from answer pool
    poss = [label2answer[random.choice(answers)] for question, answers, candis in train_raw]
    # choose 1 candidate from candidate pool
    negs = [label2answer[random.choice(candis)] for question, answers, candis in train_raw]

    # filtered word not in idx2vec
    questions_filtered = [[idx for idx in q if idx in idx2vec] for q in questions]
    poss_filtered = [[idx for idx in ans if idx in idx2vec] for ans in poss]
    negs_filtered = [[idx for idx in ans if idx in idx2vec] for ans in negs]

    # crop to seq limit
    questions_crop = [q[:q_seq_limit] + [0]*max(0, q_seq_limit - len(q)) for q in questions_filtered]
    poss_crop = [ans[:ans_seq_limit] + [0]*max(0, ans_seq_limit - len(ans)) for ans in poss_filtered]
    negs_crop = [ans[:ans_seq_limit] + [0]*max(0, ans_seq_limit - len(ans)) for ans in negs_filtered]

    # encoded, word idx to word vector
    questions_encoded = [[idx2vec[idx] for idx in q] for q in questions_crop]
    poss_encoded = [[idx2vec[idx] for idx in ans] for ans in poss_crop]
    negs_encoded = [[idx2vec[idx] for idx in ans] for ans in negs_crop]

    # make nd array
    questions_encoded=np.array(questions_encoded).astype(np.float32)
    poss_encoded=np.array(poss_encoded).astype(np.float32)
    negs_encoded=np.array(negs_encoded).astype(np.float32)
    return questions_encoded, poss_encoded, negs_encoded

def get_idx2vec_weights(wv, idx2word):
    idx2vec = {k:wv[v] for k,v in idx2word.items() if v in wv}

    # add padding embedding (all zeros) to idx2vec lookup
    idx2vec[pad_idx] = pad_embed
    return idx2vec


def prepare_data():
    import pickle
    if not os.path.isfile(insuranceQA_cache_fp):
        # no cache is found, preprocess data from scratch

        # get pretained word vector
        from gensim.models.keyedvectors import KeyedVectors
        google_news_pretrain_fp = check_exist_or_download(google_news_pretrain_embeddings_link)
        wv = KeyedVectors.load_word2vec_format(google_news_pretrain_fp, binary=True)

        # prepare insurance QA dataset
        data_zip = check_exist_or_download(insuranceQA_url)
        data_dir = unzip_data(download_dir, data_zip)

        label2answer = get_label2answer(data_dir)
        idx2word = get_idx2word(data_dir)
        idx2vec = get_idx2vec_weights(wv, idx2word)

        train_raw = get_train_raw(data_dir)
        with open(insuranceQA_cache_fp, 'wb') as handle:
            pickle.dump((train_raw, label2answer, idx2word, idx2vec), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # load from cached pickle
        with open(insuranceQA_cache_fp, 'rb') as handle:
            (train_raw, label2answer, idx2word, idx2vec) = pickle.load(handle)

    return train_raw, label2answer, idx2word, idx2vec
    # return get_filtered_crop_data(train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, wv)


def get_filtered_crop_data_training_eval(train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec, top_k_candi_limit = 6):
    ''' prepare train data to embedded word vector sequence given sequence limit for testing
        return:
            questions_encoded: np ndarray, shape
                (number samples, seq length, vector size)
            poss_encoded: same layout, sequence for positive answer
            negs_encoded: same layout, sequence for negative answer
    '''
    questions = [question for question, answers, candis in train_raw]

    # combine truth and candidate answers label,
    candi_pools = [list(answers+candis)[:top_k_candi_limit] for question, answers, candis in train_raw]
    assert all([len(pool)==top_k_candi_limit for pool in candi_pools])

    ans_count = [len(answers) for question, answers, candis in train_raw]
    assert all([c>0 for c in ans_count])

    # encode ans
    candi_pools_encoded = [[label2answer[candi_label] for candi_label in pool] for pool in candi_pools]

    # filtered word not in idx2vec
    questions_filtered = [[idx for idx in q if idx in idx2vec] for q in questions]
    candi_pools_filtered = [[[idx for idx in candi_encoded if idx in idx2vec] for candi_encoded in pool] for pool in candi_pools_encoded]

    # crop to seq limit
    questions_crop = [q[:q_seq_limit] + [0]*max(0, q_seq_limit - len(q)) for q in questions_filtered]
    candi_pools_crop = [[candi[:ans_seq_limit] + [0]*max(0, ans_seq_limit - len(candi)) for candi in pool] for pool in candi_pools_filtered]

    # encoded, word idx to word vector
    questions_encoded = [[idx2vec[idx] for idx in q] for q in questions_crop]
    candi_pools_encoded = [[[idx2vec[idx] for idx in candi] for candi in pool] for pool in candi_pools_crop]
    questions_encoded=np.array(questions_encoded).astype(np.float32)
    candi_pools_encoded=np.array(candi_pools_encoded).astype(np.float32)

    # candi_pools_encoded shape
    #    (number of sample QnA,
    #     number of candi in pool,
    #     number of sequence word idx per candi,
    #     300 word embedding for 1 word idx)
    #  e.g 10 QnA to test
    #      5 each question has 5 possible ans
    #      8 each ans has 8 words
    #      300 each word has vector size 300
    return questions_encoded, candi_pools_encoded, ans_count


if __name__ == "__main__":
    q_seq_limit = 10
    ans_seq_limit = 12
    train_raw, label2answer, idx2word, idx2vec = prepare_data()
    questions_encoded, poss_encoded, negs_encoded = get_filtered_crop_data(
        train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec)
    questions_encoded, candi_pools_encoded, ans_count = get_filtered_crop_data_training_eval(train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec)