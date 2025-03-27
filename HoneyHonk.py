import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

print("Loading HoneyHonk... (This might take a while!)")
model_name = "EleutherAI/gpt-neox-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = GPTNeoXForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("HoneyHonk is ready to chat! Type 'exit' to stop.\n")

def chat():
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("HoneyHonk: I'm out then!")
            break

        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(model.device)

        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],  
            max_length=150, 
            temperature=0.7, 
            top_p=0.9, 
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(user_input):].strip()

        print(f"HoneyHonk: {response}")

if __name__ == "__main__":
    chat()
