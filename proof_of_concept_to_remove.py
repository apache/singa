from singa import autograd
from singa import tensor
from singa import device
from singa import layer
from singa import model
from singa import opt

class DoubleLinear(layer.Layer):
    def __init__(self, a, b, c):
        super(DoubleLinear, self).__init__()
        self.l1 = layer.Linear(a,b)
        self.l2 = layer.Linear(b,c)

    def __call__(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

class MyModel(model.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = layer.Linear(4,2)
        self.bn1 = layer.BatchNorm2d(2)
        self.dl1 = DoubleLinear(2,4,2)
        self.optimizer = opt.SGD()

    def forward(self, x):
        y = self.l1(x)
        y = autograd.reshape(y, (y.shape[0], y.shape[1], 1, 1))
        y = self.bn1(y)
        y = autograd.reshape(y, (y.shape[0], y.shape[1]))
        y = self.dl1(y)
        return y

    def train_one_batch(self, x, y):
        y_ = self.forward(x)
        l = self.loss(y_, y)
        # print("loss",l)
        self.optim(l)
        return y_, l

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss):
        self.optimizer.backward_and_update(loss)


if __name__ == "__main__":
    dev = device.create_cuda_gpu_on(7)
    x = tensor.PlaceHolder((2, 4), device=dev)

    m = MyModel()
    m.on_device(dev)

    print("compile")
    # m.compile([x], is_train=True, use_graph=True, sequential=True)
    m.compile([x], is_train=True, use_graph=True, sequential=False)
    # m.compile([x], is_train=True, use_graph=False, sequential=False)

    # get params
    _ = m.get_params()
    print(_)
    # set params
    m.set_params(_)

    # get states
    print("states")
    _ = m.get_states()
    print(_)
    m.set_states(_)

    print("training")
    cx = tensor.PlaceHolder((2, 4), device=dev).gaussian(1, 1)
    cy = tensor.PlaceHolder((2, 2), device=dev).gaussian(1, 1)
    m.train_one_batch(cx, cy)
