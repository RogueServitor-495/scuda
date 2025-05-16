import torch
import numpy as np


tp = torch.bfloat16
torch.manual_seed(42)

def main():
    a = torch.randn(3, 4, dtype=tp).cuda()
    b = torch.randn(4, 5, dtype=tp).cuda()
    
    result = torch.matmul(a, b)
    
    print(result)
    



if __name__ == "__main__":
    main()