
import torch

from torch.nn import functional as F

torch.cuda.manual_seed_all(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tp = torch.bfloat16

lin = torch.nn.Linear(1024, 2048, dtype=tp)
lin.bias = None

with torch.no_grad():
    host_x = torch.randn(16, 1024, dtype=tp)
    host_y = lin(host_x)
    # print(host_y)

with torch.no_grad():
    lin = lin.cuda()
    dev_x = host_x.cuda()
    dev_y = lin(dev_x)
    print(dev_y)

print(host_y)

