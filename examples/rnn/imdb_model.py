from singa import autograd
from singa import layer
from singa import model


class IMDBModel(model.Model):

    def __init__(self,
                 hidden_size,
                 mode='lstm',
                 return_sequences=False,
                 bidirectional="False",
                 num_layers=1):
        super().__init__()
        batch_first = True
        self.lstm = layer.CudnnRNN(hidden_size=hidden_size,
                                   batch_first=batch_first,
                                   rnn_mode=mode,
                                   return_sequences=return_sequences,
                                   num_layers=1,
                                   dropout=0.9,
                                   bidirectional=bidirectional)
        self.l1 = layer.Linear(64)
        self.l2 = layer.Linear(2)

    def forward(self, x):
        y = self.lstm(x)
        y = autograd.reshape(y, (y.shape[0], -1))
        y = self.l1(y)
        y = autograd.relu(y)
        y = self.l2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_opt(self, optimizer):
        self.optimizer = optimizer
