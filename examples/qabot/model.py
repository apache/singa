from singa import autograd
from singa import layer
from singa import model
from singa import tensor
from singa import device

class QAModel(model.Model):
    def __init__(self, hidden_size, num_layers=1, rnn_mode="lstm", batch_first=True):
        super(QAModel, self).__init__()
        self.lstm_q = layer.CudnnRNN(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=True,
                                   return_sequences=False,
                                   rnn_mode=rnn_mode,
                                   batch_first=batch_first)
        self.lstm_a = layer.CudnnRNN(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=True,
                                   return_sequences=False,
                                   rnn_mode=rnn_mode,
                                   batch_first=batch_first)

    def forward(self, q, a_batch):
        q = self.lstm_q(q) # BS, Hidden*2
        a_batch = self.lstm_a(a_batch) # {2, hidden*2}
        a_pos, a_neg = autograd.split(a_batch, 0, [1,1])
        sim_pos = autograd.cossim(q, a_pos)
        sim_neg = autograd.cossim(q, a_neg)
        return sim_pos, sim_neg

    def train_one_batch(self, q, a):
        out = self.forward(q, a)
        loss = autograd.qa_lstm_loss(out[0], out[1])
        self.optimizer.backward_and_update(loss)
        return out, loss


# could not converge
class MLP(model.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = layer.Linear(500)
        self.relu = layer.ReLU()
        self.linear2 = layer.Linear(2)

    def forward(self, q, a):
        q=autograd.reshape(q, (q.shape[0], -1))
        a=autograd.reshape(a, (q.shape[0], -1))
        qa=autograd.cat([q,a], 1)
        y=self.linear1(qa)
        y=self.relu(y)
        y=self.linear2(y)
        return y

    def train_one_batch(self, q, a, y):
        out = self.forward(q, a)
        loss = autograd.softmax_cross_entropy(out, y)
        self.optimizer.backward_and_update(loss)
        return out, loss
