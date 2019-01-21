import numpy as np
from os import listdir
from os.path import isfile, join
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

data = torch.from_numpy(np.ones((4,3)).astype(np.float32))
label = torch.from_numpy(np.ones((4)).astype(np.float32))

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.w0 = torch.zeros((3,3))
        self.b = torch.zeros(3)
    def forward(self,x):
        x = torch.matmul(x,self.w0)
        x = x + self.b
        return F.relu(x)

model = Mlp()
print(model(data))
torch.onnx.export(model, data, "pytorch.onnx",verbose=True, input_names=['X'], output_names=['Y'])
