
from singa import layer
from singa import model


class CNN(model.Model):

    def __init__(self, num_classes=10, num_channels=1):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = 128
        self.dimension = 4
        self.conv1 = layer.Conv2d(num_channels, 32, 3, padding=0, activation="RELU")
        self.conv2 = layer.Conv2d(32, 64, 3, padding=0, activation="RELU")
        self.conv3 = layer.Conv2d(64, 64, 3, padding=0, activation="RELU")
        self.linear1 = layer.Linear(128)
        self.linear2 = layer.Linear(num_classes)
        self.pooling1 = layer.MaxPool2d(2, 2, padding=0)
        self.pooling2 = layer.MaxPool2d(2, 2, padding=0)
        self.pooling3 = layer.MaxPool2d(2, 2, padding=0)
        self.relu = layer.ReLU()
        self.flatten = layer.Flatten()
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
        self.sigmoid = layer

    def forward(self, x):
        y = self.conv1(x)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = self.pooling2(y)
        y = self.conv3(y)
        y = self.pooling3(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)

        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(**kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a pre-trained model.

    Returns:
        The created CNN model.
    """
    model = CNN(**kwargs)

    return model


__all__ = ['CNN', 'create_model']
