from singa import singa_wrap, autograd, tensor, device, module, opt

class MyModel(module.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l1 = autograd.Linear(2)
        self.optimizer = opt.SGD()

    def forward(self, x):
        y = self.l1(x)
        return y

    def train_one_batch(self, x, y):
        y_ = self.forward(x)
        l = self.loss(y_, y)
        print("loss",l)
        self.optim(l)
        return y_, l

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss):
        self.optimizer.backward_and_update(loss)

if __name__ == "__main__":
    PlaceHolder = tensor.Tensor
    dev = device.create_cuda_gpu()
    x = PlaceHolder((2,4), device=dev)

    m = MyModel()
    m.on_device(dev)
    # m.compile([x], is_train=True, use_graph=True, sequential=True)
    m.compile([x], is_train=True, use_graph=True, sequential=False)
    # m.compile([x], is_train=True, use_graph=False, sequential=False)

    print("compile done")

    _ = m.l1.get_params()
    print(_)
    print("get params done")

    cx = PlaceHolder((2,4), device=dev).gaussian(1,1)
    cy = PlaceHolder((2,2), device=dev).gaussian(1,1)

    m.train_one_batch(cx,cy)
    print("train done")
