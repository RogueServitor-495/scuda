import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(10)


def main():
    conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, dtype=torch.float32).to(device)
    x = torch.randn(16,3,32,32).to(device)
    print("x attached to device:", x.device)
    
    result = conv(x)
    print("result compute finished...")
    print("result is:",result)
    

if __name__ == "__main__":
    main()