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

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

class MyModel(model.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = layer.Linear(2)
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
    dev = device.create_cuda_gpu_on(0)
    x = tensor.PlaceHolder((2, 4), device=dev)

    m = MyModel()
    m.on_device(dev)

    # test function: compile
    print("function: compile")
    # m.compile([x], is_train=True, use_graph=True, sequential=True)
    m.compile([x], is_train=True, use_graph=True, sequential=False)
    # m.compile([x], is_train=True, use_graph=False, sequential=False)

    # test function get_params
    print("function: get_params")
    print(m.get_params())
    print(m.l1.get_params())

    # test function set_params
    print("function: set_params")
    params = m.get_params()
    m.set_params(params)
    params = {'MyModel.l1.W': tensor.Tensor((4, 2), device=dev).set_value(1.0),
              'MyModel.l1.b': tensor.Tensor((2,  ), device=dev).set_value(2.0),
              'MyModel.bn1.scale': tensor.Tensor((2,  ), device=dev).set_value(3.0),
              'MyModel.bn1.bias' : tensor.Tensor((2,  ), device=dev).set_value(4.0),
              'MyModel.dl1.l1.W':  tensor.Tensor((2, 4), device=dev).set_value(5.0),
              'MyModel.dl1.l1.b':  tensor.Tensor((4,  ), device=dev).set_value(6.0),
              'MyModel.dl1.l2.W':  tensor.Tensor((4, 2), device=dev).set_value(7.0),
              'MyModel.dl1.l2.b':  tensor.Tensor((2,  ), device=dev).set_value(8.0)}
    m.set_params(params)
    print(m.get_params())

    # test function get_states
    print("function: get_states")
    print(m.get_states())
    print(m.bn1.get_states())

    # test function set_states
    print("function: set_states")
    states = m.get_states()
    m.set_states(states)
    states = {'MyModel.l1.W': tensor.Tensor((4, 2), device=dev).set_value(9.0),
              'MyModel.l1.b': tensor.Tensor((2,  ), device=dev).set_value(10.0),
              'MyModel.bn1.scale': tensor.Tensor((2,  ), device=dev).set_value(11.0),
              'MyModel.bn1.bias' : tensor.Tensor((2,  ), device=dev).set_value(12.0),
              'MyModel.bn1.running_mean':  tensor.Tensor((2,  ), device=dev).set_value(13.0),
              'MyModel.bn1.running_var' :  tensor.Tensor((2,  ), device=dev).set_value(14.0),
              'MyModel.dl1.l1.W':  tensor.Tensor((2, 4), device=dev).set_value(15.0),
              'MyModel.dl1.l1.b':  tensor.Tensor((4,  ), device=dev).set_value(16.0),
              'MyModel.dl1.l2.W':  tensor.Tensor((4, 2), device=dev).set_value(17.0),
              'MyModel.dl1.l2.b':  tensor.Tensor((2,  ), device=dev).set_value(18.0)}
    m.set_states(states)
    print(m.get_states())

    print("training")
    cx = tensor.PlaceHolder((2, 4), device=dev).gaussian(1, 1)
    cy = tensor.PlaceHolder((2, 2), device=dev).gaussian(1, 1)
    m.train_one_batch(cx, cy)
