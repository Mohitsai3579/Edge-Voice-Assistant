"""Stage-by-stage latency and memory benchmark for Edge Voice Assistant.

Usage: python benchmark.py sample.wav --runs 10
Requires: psutil (pip install psutil). Uses the same models as app.py.
"""
import argparse, os, statistics, time
import psutil, torch
from datasets import load_dataset
from transformers import pipeline, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

p = argparse.ArgumentParser()
p.add_argument("wav")
p.add_argument("--runs", type=int, default=10)
a = p.parse_args()

proc = psutil.Process(os.getpid())
peak = 0
def note_mem():
    global peak
    peak = max(peak, proc.memory_info().rss)

stt = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
llm = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M-Instruct", device="cpu", torch_dtype=torch.float32)
tok = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
voc = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
spk = torch.tensor(load_dataset("regisss/cmu-arctic-xvectors", split="validation")[7306]["xvector"]).unsqueeze(0)
note_mem()

def run_stt():
    return stt(a.wav)["text"].strip()

def run_llm(text):
    prompt = llm.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return llm(prompt, max_new_tokens=50, do_sample=True, temperature=0.6)[0]["generated_text"].split("assistant\n")[-1].strip()

def run_tts(text):
    ids = tok(text=text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return tts.generate_speech(ids, spk, vocoder=voc)

text = run_stt(); reply = run_llm(text); run_tts(reply)  # warm-up, not timed
res = {"stt": [], "llm": [], "tts": []}
for _ in range(a.runs):
    t = time.perf_counter(); text = run_stt();    res["stt"].append(time.perf_counter() - t); note_mem()
    t = time.perf_counter(); reply = run_llm(text); res["llm"].append(time.perf_counter() - t); note_mem()
    t = time.perf_counter(); run_tts(reply);      res["tts"].append(time.perf_counter() - t); note_mem()

def p95(xs):
    return sorted(xs)[max(0, int(round(0.95 * len(xs))) - 1)]

print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | runs {a.runs}")
print("| stage | median s | p95 s |\n|---|---|---|")
for k, v in res.items():
    print(f"| {k} | {statistics.median(v):.2f} | {p95(v):.2f} |")
print(f"| total (sum of medians) | {sum(statistics.median(v) for v in res.values()):.2f} | |")
print(f"peak RSS: {peak / 1e9:.2f} GB")
