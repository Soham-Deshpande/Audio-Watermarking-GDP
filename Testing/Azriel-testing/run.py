import os
import sys
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Paths to import from root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(root_dir)
# Import your new modules 
from attack import PsychoacousticPGD
try:
    from functions import signalToNoise # Importing Soham's helper
except ImportError:
    print("Warning: functions.py not found. Skipping SNR calculation.")

def main():
    # Configuration
    # Update this path to point if needed
    INPUT_FILE = os.path.join(root_dir, "Testing/Connor-Testing/samples/sample.flac")
    OUTPUT_FILE = os.path.join(current_dir, "adversarial_example.flac")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model 
    print("Loading Wav2Vec2 model...")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

    # Load Audio
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    audio, sr = librosa.load(INPUT_FILE, sr=16000)
    print(f"Loaded audio: {len(audio)/sr:.2f} seconds")

    # Run the Attack
    attacker = PsychoacousticPGD(model, processor, device=device)
    
    # Tune these parameters! 
    # epsilon=0.01 is quieter, epsilon=0.05 is stronger
    adv_audio = attacker.generate(audio, sr, epsilon=0.02, alpha=0.002, num_iter=50)

    # Save Result
    sf.write(OUTPUT_FILE, adv_audio, sr)
    print(f"Success! Saved adversarial audio to: {OUTPUT_FILE}")

    # Evaluate
    # Calculate Signal-to-Noise Ratio using the group's function
    noise = adv_audio - audio
    try:
        snr = signalToNoise(audio, noise)
        print(f"Signal-to-Noise Ratio (SNR): {snr:.2f} dB (Higher is better quality)")
    except NameError:
        pass

if __name__ == "__main__":

    main()
