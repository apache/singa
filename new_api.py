from singa import autograd
from singa import layer
from singa import tensor
from singa import device
from singa import model
from singa import opt


class MyModel(model.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = layer.Linear(2)
        self.optimizer = opt.SGD()

    def forward(self, x):
        y = self.l1(x)
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
    dev = device.create_cuda_gpu()
    x = tensor.PlaceHolder((2, 4), device=dev)

    m = MyModel()
    m.on_device(dev)
    # m.compile([x], is_train=True, use_graph=True, sequential=True)
    m.compile([x], is_train=True, use_graph=True, sequential=False)
    # m.compile([x], is_train=True, use_graph=False, sequential=False)
    print("compile done")

    params = m.get_params()
    print(params)
    print("get params done")

    params = {'MyModel.l1.W': tensor.Tensor((4, 2), device=dev).set_value(3.0),
              'MyModel.l1.b': tensor.Tensor((2,  ), device=dev).set_value(4.0)}
    # params = {'l1': {'W': tensor.Tensor((4, 2), device=dev).set_value(3.0),
    #                  'b': tensor.Tensor((2,  ), device=dev).set_value(4.0)}}
    m.set_params(**params)
    params = m.get_params()
    print(params)

    cx = tensor.PlaceHolder((2, 4), device=dev).gaussian(1, 1)
    cy = tensor.PlaceHolder((4, ), device=dev).gaussian(1, 1)

    print("start training")
    m.train_one_batch(cx, cy)
    print("train done")
