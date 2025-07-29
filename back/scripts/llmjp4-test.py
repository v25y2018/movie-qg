import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3.1-13b-instruct4")
model = AutoModelForCausalLM.from_pretrained("llm-jp/llm-jp-3.1-13b-instruct4", device_map="auto", torch_dtype=torch.bfloat16)
chat = [
    {"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
    {"role": "user", "content": "自然言語処理とは何か"},
]
tokenized_input = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        tokenized_input,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.05,
    )[0]
print(tokenizer.decode(output))

