"""
Run: python chatbot/try_qwen.py


"""
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# the model lives in <project>/model/, one folder up from this script
MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "Qwen3-0.6B-Base"
MAX_NEW_TOKENS = 120
TEMPERATURE = 0.8

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Loading {MODEL_DIR} on {device}...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
print('Ready. Type a prompt ("\\n" = line break, "quit" to exit).\n')

while True:
    try:
        prompt = input(">>> ")
    except (EOFError, KeyboardInterrupt):
        break
    if prompt.strip().lower() in {"quit", "exit"}:
        break
    if not prompt.strip():
        continue

    prompt = prompt.replace("\\n", "\n")
    inputs = tok(prompt, return_tensors="pt").to(device)
    print(prompt, end="", flush=True)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                       temperature=TEMPERATURE, top_p=0.95, repetition_penalty=1.1,
                       pad_token_id=tok.eos_token_id, streamer=streamer)
    print()
