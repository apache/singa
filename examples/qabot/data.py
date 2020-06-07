from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random

def load_vocabulary(vocab_path, label_path):
    id_to_word = {}
    with open(vocab_path,'rb') as f:
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
                label_to_ans_text[label] = [id_to_word[t] for t in answer.split(' ')]
    return id_to_word, label_to_ans, label_to_ans_text

def parse_file(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            poss = [label_to_ans_text[t] for t in d[2].split(' ')] # ground-truth
            negs = [label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]] # candidate-pool without ground-truth
            for pos in poss:
                data.append((q, pos, negs))
    return data

def parse_test_file(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines[12:]:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')] # question
            poss = [t for t in d[2].split(' ')] # ground-truth
            cands = [t for t in d[3].split(' ')] # candidate-pool
            data.append((q, poss, cands))
    return data

def load_data(embed_path='GoogleNews-vectors-negative300.bin'):
    id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
    train_data = parse_file('./V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded', id_to_word, label_to_ans_text)
    print("loaded training set ", len(train_data))

    qa_tuples = generate_qa_triplets(train_data) # (q, a+, a-)

    word2vec = KeyedVectors.load_word2vec_format(embed_path, binary=True)
    print("successfully loaded word2vec model")

    training = qa_tuples_to_naive_training_format(word2vec, qa_tuples) # (q, a-, 0) (q, a+, 1)
    return training

def load_data_small2(embed_path='GoogleNews-vectors-negative300.bin'):
    id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
    train_data = parse_file('./V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded', id_to_word, label_to_ans_text)
    train_data_small = train_data[:10]

    # test_data = parse_test_file('InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded', id_to_word, label_to_ans_text)
    # test_data = test_data[:10]

    qa_tuples = generate_qa_triplets(train_data_small) # (q, a+, a-)

    # training = qa_tuples_to_naive_training_format(word2vec, qa_tuples) # (q, a-, 0) (q, a+, 1)
    return qa_tuples

def load_data_small(embed_path='GoogleNews-vectors-negative300.bin'):
    id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
    train_data = parse_file('./V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded', id_to_word, label_to_ans_text)
    test_data = parse_test_file('InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded', id_to_word, label_to_ans_text)
    train_data_small = train_data[:10]
    test_data = test_data[:10]

    qa_tuples = generate_qa_triplets(train_data_small) # (q, a+, a-)

    word2vec = KeyedVectors.load_word2vec_format(embed_path, binary=True)
    print("successfully loaded word2vec model")

    training = qa_tuples_to_naive_training_format(word2vec, qa_tuples) # (q, a-, 0) (q, a+, 1)
    return training


def words_text_to_fixed_seqlen_vec(word2vec, words,sentence_length=10):
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

def generate_qa_triplets(data, num_negs=10):
    tuples = []
    for (q, a_pos, a_negs) in data:
        for i in range(num_negs):
            tpl = (q, a_pos, random.choice(a_negs))
            tuples.append(tpl)
    return tuples

def qa_tuples_to_naive_training_format(wv, tuples):
    training = []
    q_len_limit = 10
    a_len_limit = 100
    for tpl in tuples:
        q, a_pos, a_neg = tpl
        q_vec = words_text_to_fixed_seqlen_vec(wv, q,q_len_limit)
        training.append((q_vec, words_text_to_fixed_seqlen_vec(wv, a_pos,a_len_limit), 1))
        training.append((q_vec, words_text_to_fixed_seqlen_vec(wv, a_neg,a_len_limit), 0))
    return training


if __name__ == "__main__":
    train = load_data_small() # num samples * (q, a, label)
    print(train[0])
    print(len(train))
