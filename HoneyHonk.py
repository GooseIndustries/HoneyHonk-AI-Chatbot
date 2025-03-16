import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

print("Loading HoneyHonk... (This might take a while!)")
model_name = "EleutherAI/gpt-neox-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoXForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("HoneyHonk is ready to chat! Type 'exit' to stop.\n")

def chat():
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("HoneyHonk: Bzzz! Catch you later! 🐝💨")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs["input_ids"], 
            max_length=150, 
            temperature=0.7, 
            top_p=0.9, 
            top_k=50,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"HoneyHonk: {response}")

if __name__ == "__main__":
    chat()
