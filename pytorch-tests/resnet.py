import torch
from torchvision import models
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
model.eval()  


random_input = torch.rand(1, 3, 224, 224).to(device) 


for i in range(10):
    with torch.no_grad():
        output = model(random_input)
        print("Output shape:", output.shape)  
        print("Sample output values:", output[0, :5])  

start = time.time() 

reps = 100
for i in range(reps):
    
    random_input = torch.rand(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(random_input)
        print("Output shape:", output.shape) 
        print("Sample output values:", output[0, :5])  
        
end = time.time()
print("time cost: ", (end - start)/reps)
        





