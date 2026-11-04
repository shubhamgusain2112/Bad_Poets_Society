"""

Run:  python chatbot/chat_qwen.py
Commands- "reset" clears the conversation, "quit" exits.


"""
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StopStringCriteria

# the model lives in <project>/model/, one folder up from this script
MODEL_DIR = Path(__file__).resolve().parent.parent / "model" / "Qwen3-0.6B-Base"
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7

INTRO = (
    "The following is a conversation between a curious user and a helpful, friendly assistant. "
    "The assistant gives short, clear, correct answers.\n\n"
    "User: Hi! Who are you?\n"
    "Assistant: Hello! I'm an AI assistant. Ask me anything and I'll do my best to help.\n\n"
    "User: What is 12 times 3?\n"
    "Assistant: 12 times 3 is 36.\n\n"
)
STOP = "\nUser:"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"Loading {MODEL_DIR} on {device}...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(device).eval()
# built once: constructing it scans the whole vocabulary and takes over a second
stop_criteria = StoppingCriteriaList([StopStringCriteria(tok, [STOP])])
print('Ready. Type a message ("reset" to start over, "quit" to exit).\n')

transcript = INTRO
while True:
    try:
        user = input("You: ").strip().lstrip("﻿")
    except (EOFError, KeyboardInterrupt):
        break
    if user.lower() in {"quit", "exit"}:
        break
    if user.lower() == "reset":
        transcript = INTRO
        print("(conversation cleared)\n")
        continue
    if not user:
        continue

    transcript += f"User: {user}\nAssistant:"
    inputs = tok(transcript, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                             temperature=TEMPERATURE, top_p=0.9, repetition_penalty=1.1,
                             stopping_criteria=stop_criteria, pad_token_id=tok.eos_token_id)
    elapsed = time.perf_counter() - start
    n_new = out.shape[1] - inputs["input_ids"].shape[1]
    reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    reply = reply.split(STOP)[0].strip()

    print(f"Qwen: {reply}")
    print(f"({n_new} tokens in {elapsed:.1f}s, {n_new / elapsed:.1f} tok/s)\n")
    transcript += f" {reply}\n\n"
