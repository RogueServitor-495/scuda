from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers.models.qwen3 import modeling_qwen3 as qwenmod


model_path = "/mnt-client/Qwen3-0.6B"

def register_hook_print(module, input, output):
    # pass
#     print("module : {module}")
    if hasattr(module,"name"):
        name = module.name
    else:
        name = module    
    # try:
    #     device = next(module.parameters()).device
    # except StopIteration:
    #     device = input[0].device if isinstance(input, tuple) else input.device
    print( "*"*10 + f"executing module {name} " + "*"*10)


        
def register_hooks_recursive(module, parent_name=""):
    for name, submodule in module.named_modules():
        submodule.register_forward_hook(register_hook_print)
       
 

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

print(model)

register_hooks_recursive(model)


# prepare the model input
prompt = "please介绍一下你自己."
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


def manual_gen(input_ids):
    """
    不使用transformers的generate方法，手动循环生成token。缺点：每次生成token，似乎都需要把input结果传输至client
    """

    max_tokens = 50

    # # 手动forward
    with torch.no_grad():
        outputs = model(input_ids = input_ids, attention_mask = None, use_cache=True)

    logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    past = outputs.past_key_values
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated_ids = [next_id]

    for _ in range(max_tokens):
        with torch.no_grad():
            last_id = generated_ids[-1].to(model.device)
            outputs = model(input_ids=last_id, past_key_values=past, use_cache=True)
        logits = outputs.logits
        past = outputs.past_key_values
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_id)

    genTokens = torch.cat([input_ids] + generated_ids, dim=-1)
    
    texts =  tokenizer.batch_decode(genTokens, skip_special_tokens=True)

    print("Generated text:   ", texts)

    print("end")
    

def integrated_gen(input_ids):
    


    # # 直接generate
    # conduct text completion
    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=32768
    # )

    # 手动embedding后generate
    generated_ids = model.generate(
        input_ids = input_ids,
        attention_mask = None,
        max_new_tokens=32768
    )


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
    


if __name__ == "__main__":
    manual_gen(input_ids)
