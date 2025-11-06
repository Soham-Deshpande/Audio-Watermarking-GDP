# Main file for streamlit application

### Imports ###
import streamlit as st 
import pandas as pd 
import numpy as np
from pydub import AudioSegment
import io
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torch.nn.functional as F


###### Webpage #######

st.set_page_config(page_title="Audio Upload & FGSM Attack", layout="wide")

# -------------------------
# Cache the model
# -------------------------
@st.cache_resource(show_spinner=False)
def load_surrogate_model(model_name="facebook/wav2vec2-base-960h", device_str=None):
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    return processor, model, device

W2V_DICT = {
    '-': 0, '|': 1, 'E': 2, 'T': 3, 'A': 4, 'O': 5, 'N': 6, 'I': 7, 'H': 8, 'S': 9,
    'R': 10, 'D': 11, 'L': 12, 'U': 13, 'M': 14, 'W': 15, 'C': 16, 'F': 17, 'G': 18,
    'Y': 19, 'P': 20, 'B': 21, 'V': 22, 'K': 23, "'": 24, 'X': 25, 'J': 26, 'Q': 27, 'Z': 28
}

# -------------------------
# Transcription helper
# -------------------------
def transcribe_surrogate(arr, sr, processor, model, device):
    inputs = processor(arr, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(pred_ids)[0]
    return transcript.upper().strip()

# -------------------------
# FGSM function (untargeted)
# -------------------------
def FGSM_target(audio_arr, sr, processor, model, device, encoded_transcription, eps=0.002):
    """
    Single-step FGSM that *maximizes* the CTC loss w.r.t the surrogate transcription encoding.
    Returns numpy array (float32) of adv audio, clamped to [-1, 1].
    """
    inputs = processor(audio_arr, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)   # shape (1, T)
    input_values = input_values.clone()
    input_values.requires_grad = True

    logits = model(input_values).logits  # (1, seq_len, vocab)
    target = encoded_transcription.to(device).long()
    logits_len = torch.tensor([logits.shape[1]], dtype=torch.long).to(device)
    target_len = torch.tensor([len(encoded_transcription)], dtype=torch.long).to(device)

    logits_t = logits.transpose(0, 1)  # (seq_len, 1, vocab)
    loss = F.ctc_loss(logits_t, target.unsqueeze(0), logits_len, target_len, blank=0, reduction='mean')
    loss.backward()
    sign = input_values.grad.sign()
    adv = input_values + eps * sign
    adv = torch.clamp(adv, -1.0, 1.0).detach().cpu().numpy()[0].astype(np.float32)
    return adv

# -------------------------
# Visualisation
# -------------------------
def plot_waveform(ax, samples, sr, title="Waveform"):
    duration = len(samples) / sr
    t = np.linspace(0, duration, num=len(samples))
    ax.plot(t, samples)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

def plot_spectrogram(ax, samples, sr, title="Spectrogram"):
    Pxx, freqs, bins, im = ax.specgram(samples, Fs=sr, NFFT=1024, noverlap=512)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)

# -------------------------
# Sidebar navigation
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Visualize", "FGSM Attack", "About Audio Watermarking"])

# -------------------------
# Page 1: Upload & Visualize
# -------------------------
if page == "Upload & Visualize":
    st.title("Upload & Visualize Audio")
    uploaded = st.file_uploader("Upload audio (wav, mp3, flac, ogg)", type=["wav","mp3","flac","ogg"])
    if uploaded is not None:
        with st.spinner("Loading audio..."):
            audio_bytes = uploaded.read()
            try:
                audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            except Exception as e:
                st.error(f"Could not load audio: {e}")
                raise st.stop()

        st.audio(audio_bytes)  # playback
        st.write(f"Sample rate: {sr} Hz — Duration: {len(audio_np)/sr:.2f} s")

        # waveform
        fig_wf, ax_wf = plt.subplots(figsize=(10, 3))
        plot_waveform(ax_wf, audio_np, sr, title="Original Waveform")
        st.pyplot(fig_wf)

        # spectrogram
        fig_sp, ax_sp = plt.subplots(figsize=(10, 4))
        plot_spectrogram(ax_sp, audio_np, sr, title="Original Spectrogram")
        st.pyplot(fig_sp)

# -------------------------
# Page 2: FGSM Attack
# -------------------------
elif page == "FGSM Attack":
    st.title("FGSM Attack")
    st.markdown(
            "Upload an audio file, choose FGSM parameters, then run:"
        " create a noisy copy, transcribe with the Wav2Vec2 model, compute FGSM perturbation "
        "and show audio + transcripts for original, noisy, and adversarial outputs."
    )

    uploaded = st.file_uploader("Upload audio to attack (wav, mp3, flac, ogg)", type=["wav","mp3","flac","ogg"])
    st.markdown("### FGSM parameters")
    eps = st.number_input("EPS (FGSM step size)", value=0.002, step=0.0005, format="%.6f")
    iters = st.slider("Iterations (apply FGSM iteratively)", min_value=1, max_value=20, value=1)
    noise_std = st.number_input("Additive noise std (for 'noisy' version)", value=0.001, step=0.0005, format="%.6f")
    device_choice = st.selectbox("Device for  model", options=["auto","cpu"], index=0)
    run_button = st.button("Run FGSM attack")

    if uploaded is not None:
        audio_bytes = uploaded.read()
        try:
            audio_np, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            audio_np = audio_np.astype(np.float32)
        except Exception as e:
            st.error(f"Could not load audio: {e}")
            st.stop()

        st.write(f"Loaded audio — sr: {sr} Hz, duration: {len(audio_np)/sr:.2f} s")

        # show original waveform & spectrogram
        c1, c2 = st.columns(2)
        with c1:
            fig_wf, ax_wf = plt.subplots(figsize=(8,2.5))
            plot_waveform(ax_wf, audio_np, sr, title="Original Waveform")
            st.pyplot(fig_wf)
        with c2:
            fig_sp, ax_sp = plt.subplots(figsize=(8,3))
            plot_spectrogram(ax_sp, audio_np, sr, title="Original Spectrogram")
            st.pyplot(fig_sp)

        if run_button:
            device_str = None if device_choice=="auto" else "cpu"
            with st.spinner("Loading model (may take time on first run)..."):
                processor, model, device = load_surrogate_model(device_str=device_str)

            rng = np.random.default_rng(seed=42)
            noise = rng.normal(0, noise_std, size=len(audio_np)).astype(np.float32)
            audio_noise = audio_np + noise
            audio_noise = np.clip(audio_noise, -1.0, 1.0).astype(np.float32)

            st.info("Transcribing original and noisy audio with model...")
            orig_transcript = transcribe_surrogate(audio_np, sr, processor, model, device)
            noisy_transcript = transcribe_surrogate(audio_noise, sr, processor, model, device)

            st.markdown("**Original transcript**")
            st.code(orig_transcript)
            st.markdown("**Noisy transcript**")
            st.code(noisy_transcript)

            chars = list(orig_transcript)
            encoded_chars = [W2V_DICT[c] for c in chars if c in W2V_DICT]
            if len(encoded_chars) == 0:
                st.warning("Could not encode original transcript with the limited dictionary — FGSM may fail.")
            encoded_transcription = torch.tensor(encoded_chars, dtype=torch.long)

            # Run iterative FGSM (apply FGSM_target repeatedly)
            adv = audio_np.copy()
            progress_text = st.empty()
            for i in range(iters):
                progress_text.text(f"Iteration {i+1}/{iters} ...")
                try:
                    adv = FGSM_target(adv, sr, processor, model, device, encoded_transcription, eps=eps)
                except Exception as e:
                    st.error(f"FGSM step failed: {e}")
                    break
            progress_text.empty()
            adv = np.clip(adv, -1.0, 1.0).astype(np.float32)
            adv_transcript = transcribe_surrogate(adv, sr, processor, model, device)
            st.success("FGSM attack complete — results below")

            # --- Playback and downloads ---
            st.write("### Playback")

            def to_wav_bytes(arr, sr):
                buf = io.BytesIO()
                sf.write(buf, arr, sr, format='WAV')
                buf.seek(0)
                return buf.read()

            orig_bytes = to_wav_bytes(audio_np, sr)
            noisy_bytes = to_wav_bytes(audio_noise, sr)
            adv_bytes = to_wav_bytes(adv, sr)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Original**")
                st.audio(orig_bytes)
                st.download_button("Download Original", data=orig_bytes, file_name="original.wav", mime="audio/wav")
            with col2:
                st.markdown("**Noisy**")
                st.audio(noisy_bytes)
                st.download_button("Download Noisy", data=noisy_bytes, file_name="noisy.wav", mime="audio/wav")
            with col3:
                st.markdown("**Adversarial (FGSM)**")
                st.audio(adv_bytes)
                st.download_button("Download FGSM", data=adv_bytes, file_name="fgsm.wav", mime="audio/wav")

            # Transcripts
            st.write("### Transcripts")
            st.markdown("- **Original:**")
            st.code(orig_transcript)
            st.markdown("- **Noisy:**")
            st.code(noisy_transcript)
            st.markdown("- **Adversarial (FGSM):**")
            st.code(adv_transcript)

            # Waveform + spectrogram for noisy and adv
            st.write("### Visualizations")
            r1, r2 = st.columns(2)
            with r1:
                fig_nw, ax_nw = plt.subplots(figsize=(9,3))
                plot_waveform(ax_nw, audio_noise, sr, title="Noisy Waveform")
                st.pyplot(fig_nw)
                fig_ns, ax_ns = plt.subplots(figsize=(9,3))
                plot_spectrogram(ax_ns, audio_noise, sr, title="Noisy Spectrogram")
                st.pyplot(fig_ns)
            with r2:
                fig_aw, ax_aw = plt.subplots(figsize=(9,3))
                plot_waveform(ax_aw, adv, sr, title="FGSM Waveform")
                st.pyplot(fig_aw)
                fig_as, ax_as = plt.subplots(figsize=(9,3))
                plot_spectrogram(ax_as, adv, sr, title="FGSM Spectrogram")
                st.pyplot(fig_as)

# -------------- PAGE 3: ABOUT AUDIO WATERMARKING --------------
elif page == "About Audio Watermarking":
    st.title("About Audio Watermarking")

    st.markdown("""
    **Audio watermarking** is a technique used to embed information into an audio signal
    in a way that is **imperceptible to the listener** but can be **detected or extracted** later.
    """)




