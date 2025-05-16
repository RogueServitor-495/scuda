import torch
import torch.nn.functional as F

def test_softmax_basic():
    """测试基本的Softmax功能"""

    torch.manual_seed(42)
    
    tp = torch.bfloat16
    
    x = torch.tensor([[1.0, 2.0, 3.0], 
                     [1.0, 2.0, 1.0]],dtype=tp).cuda()

    output = F.softmax(x, dim=1)
    
    print(output)

test_softmax_basic()