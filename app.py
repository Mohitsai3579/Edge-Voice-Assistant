import gradio as gr
import torch
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

# --- 1. THE BOT CLASS (Logic) ---
class ResumeVoiceBot:
    def __init__(self):
        print("⚙️ Loading Models... (This runs only once)")
        self.device = "cpu"

        # Ears (Whisper)
        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=self.device
        )

        # Brain (SmolLM2)
        self.llm_pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct",
            device=self.device,
            torch_dtype=torch.float32
        )

        # Mouth (SpeechT5)
        self.tts_processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        self.speaker_embeddings = torch.tensor(load_dataset("regisss/cmu-arctic-xvectors", split="validation")[7306]["xvector"]).unsqueeze(0).to(self.device)
        
        print("✅ Models Loaded!")

    def process_conversation(self, audio_path):
        """
        1. Takes audio file path from UI
        2. Transcribes (STT)
        3. Generates Reply (LLM)
        4. Synthesizes Speech (TTS)
        """
        if audio_path is None:
            return "Please record something!", None

        # --- A. STT (Transcribe) ---
        try:
            text = self.stt_pipe(audio_path)["text"].strip()
        except Exception as e:
            return f"Error reading audio: {e}", None

        # --- BUG FIX: Hallucination Filter ---
        # If Whisper hears silence, it often outputs these phrases. We block them.
        hallucinations = ["end of the video", "thanks for watching", "subscribe", "subtitles"]
        if not text or len(text) < 2 or any(h in text.lower() for h in hallucinations):
            return "(Silence or Background Noise detected - Try Speaking Louder)", None

        print(f"User said: {text}")

        # --- B. LLM (Think) ---
        messages = [{"role": "user", "content": text}]
        prompt = self.llm_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response = self.llm_pipe(
            prompt, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.6
        )[0]['generated_text']
        
        bot_reply = response.split("assistant\n")[-1].strip()
        print(f"Bot reply: {bot_reply}")

        # --- C. TTS (Speak) ---
        inputs = self.tts_processor(text=bot_reply, return_tensors="pt").to(self.device)
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"], 
                self.speaker_embeddings, 
                vocoder=self.vocoder
            )
        
        # Save audio to a temporary file for the UI to play
        output_path = "response.wav"
        wav.write(output_path, rate=16000, data=speech.cpu().numpy())
        
        return f"👤 You: {text}\n🤖 Bot: {bot_reply}", output_path

# --- 2. INITIALIZE BOT ---
bot = ResumeVoiceBot()

# --- 3. THE UI (Gradio) ---
with gr.Blocks(title="AI Voice Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Edge AI Voice Assistant")
    gr.Markdown("Runs 100% locally on CPU using Whisper, SmolLM2, and SpeechT5.")
    
    with gr.Row():
        with gr.Column():
            # Input: Microphone
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Your Voice")
            submit_btn = gr.Button("Talk to Bot", variant="primary")
        
        with gr.Column():
            # Output: Text Log + Audio Response
            chat_log = gr.Textbox(label="Conversation Log")
            audio_output = gr.Audio(label="Bot Response", type="filepath", autoplay=True)

    # Link the button to the function
    submit_btn.click(
        fn=bot.process_conversation,
        inputs=audio_input,
        outputs=[chat_log, audio_output]
    )

# Launch the Web App
if __name__ == "__main__":
    demo.launch(share=True)