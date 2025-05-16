from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/mnt-client/Qwen3-0.6B"

def register_hook_print(module, input, output):
    print("module : {module}")
    # if hasattr(module,"name"):
    #     name = module.name
    # else:
    #     name = module    
    # try:
    #     device = next(module.parameters()).device
    # except StopIteration:
    #     device = input[0].device if isinstance(input, tuple) else input.device
    # print( "*"*20 + f"executing module on {device} : {name}" + "*"*20)

    
    # if isinstance(module, torch.nn.Linear) and module.out_features==151936:
        
    #     print("[Output] token generated...")

def hook_before(module, input):
    if isinstance(module, torch.nn.Embedding):
        print("hook at embedding...")
        


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

print(model)

for name,module in model.named_modules():
    # print(f"name : {name}, module : {module}")
    if len(list(module.children())) == 0:
        print(f"module : {module}")
        module._forward_hooks[name] = register_hook_print
        module._forward_pre_hooks[name] = hook_before
    

# for name,modules in model.named_modules():
#     print(f" layer : {name}, type : {modules}")

# prepare the model input
prompt = "介绍一下你自己."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_ids = model_inputs["input_ids"]
attention_mask = model_inputs["attention_mask"]



with torch.no_grad():
    outputs = model(input_ids = input_ids, attention_mask = attention_mask)

logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]

# 取最后一个 token 的预测（即下一个 token）
next_token_logits = logits[0, -1]  # shape: [vocab_size]
predicted_token_id = torch.argmax(next_token_logits).item()

# 解码生成 token
predicted_token = tokenizer.decode(predicted_token_id)
print("Predicted next token:", predicted_token)




# output_ids = hidden_states[0][len(model_inputs.input_ids[0]):].tolist()
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
