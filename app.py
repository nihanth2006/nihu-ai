import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a lightweight model
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat(user_input, history=[]):
    # System prompt
    system_prompt = "You are Nihu AI, a friendly assistant created by Nihanth. If asked your name, always reply 'I am Nihanth.'"
    prompt = system_prompt + "\nHuman: " + user_input + "\nAI:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("AI:")[-1].strip()

    history.append((user_input, response))
    return history, history

# Launch Gradio chat UI
iface = gr.ChatInterface(fn=chat)
iface.launch()
