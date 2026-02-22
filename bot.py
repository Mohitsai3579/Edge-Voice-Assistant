import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class CPUBot:
    def __init__(self):
        print("⚙️  Initializing CPU-Optimized Bot...")
        
        # 1. Force CPU Device
        self.device = "cpu"
        
        # 2. Initialize STT (Ears) - Whisper Tiny
        # "tiny" is the fastest model, perfect for CPU
        print(" Loading Ears (Whisper)...")
        self.stt_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device=self.device
        )

        # 3. Initialize LLM (Brain) - SmolLM2-360M
        # We use the 360M version instead of 1.7B so it runs fast on CPU
        print(" Loading Brain (SmolLM2-360M)...")
        self.llm_pipe = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-360M-Instruct",
            device=self.device,
            torch_dtype=torch.float32 # CPU works best with float32
        )

        # 4. Initialize TTS (Mouth) - SpeechT5
        print(" Loading Mouth (SpeechT5)...")
        self.tts_processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        
        # Load a default speaker embedding (voice)
        # Note: This might download a small dataset on first run
        # Use this updated parquet version that works with new libraries
        embeddings_dataset = load_dataset("regisss/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
        
        print("\n Bot is ready! Press Ctrl+C to stop.")

    def record_audio(self, duration=5, samplerate=16000):
        """Records audio from the microphone."""
        print("\n🎤 Listening... (Speak now)")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        return recording.squeeze()

    def speak(self, text):
        """Converts text to speech and plays it."""
        if not text: return
        print(f"🤖 Speaking: {text}")
        
        inputs = self.tts_processor(text=text, return_tensors="pt").to(self.device)
        
        # Generate audio
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"], 
                self.speaker_embeddings, 
                vocoder=self.vocoder
            )

        # Play audio
        sd.play(speech.cpu().numpy(), samplerate=16000)
        sd.wait()

    def run(self):
        """Main Loop: Listen -> Think -> Speak"""
        print("------------------------------------------------")
        print("  Starting Conversation Loop")
        print("  (Adjust 'duration' in code if 4s is too short)")
        print("------------------------------------------------")
        
        while True:
            try:
                # 1. Listen
                audio_data = self.record_audio(duration=4) # Record for 4 seconds
                
                # 2. Transcribe (STT)
                try:
                    result = self.stt_pipe(audio_data)["text"]
                except Exception:
                    continue # Skip if audio was empty/error

                if len(result.strip()) == 0:
                    print("... (Silence detected)")
                    continue
                
                print(f"👤 You said: {result}")

                # 3. Think (LLM)
                # Chat template for SmolLM
                messages = [{"role": "user", "content": result}]
                prompt = self.llm_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Generate response (Limited to 40 tokens for speed)
                response = self.llm_pipe(
                    prompt, 
                    max_new_tokens=40, 
                    do_sample=True, 
                    temperature=0.6,
                    top_k=50
                )[0]['generated_text']
                
                # Extract just the assistant's part
                bot_reply = response.split("assistant\n")[-1].strip()

                # 4. Speak (TTS)
                self.speak(bot_reply)

            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                # Print error but keep running
                print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    bot = CPUBot()
    bot.run()