import torch
import time

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

torch.manual_seed(10)

def main():
    tensor_a = torch.randn(1,3,2,2)
    print("tensor a created on host")
    tensor_b = torch.randn(1,3,2,2)
    print("tensor b created on host")
    
    d_a = tensor_a.to(device)
    print("tensor a attached to device:",device)
    print(tensor_a)
    print(d_a)
    
    d_b = tensor_b.to(device)
    print("tensor b attached to device:",device)
    print(tensor_b)
    print(d_b)
    
    result = torch.mul(d_a,d_b)
    print("tensor add completed...")
    detach = result.detach().cpu()
    print("tensor add result:",detach)
    print(result)

if __name__ == "__main__":
    print("use device:",device)
    start = time.time()
    main()
    end = time.time()
    print(f"time cost:{end - start}")