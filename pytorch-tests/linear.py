
import torch

model = torch.nn.Linear(128, 64).cuda()
x = torch.randn(32, 128).cuda()
y = model(x)
print(y)
