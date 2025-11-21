import numpy as np
import librosa

def compute_masking_threshold(audio, sr, window_size=2048, hop_size=512):
    """
    Computes the psychoacoustic masking threshold of an audio signal.
    Returns a frequency-domain curve where noise is inaudible.
    """
    # 1. Compute the Short-Time Fourier Transform (STFT)
    # This breaks audio into frequency components over time
    stft = librosa.stft(audio, n_fft=window_size, hop_length=hop_size)
    magnitude = np.abs(stft)
    
    # 2. Estimate Masking Threshold
    # In a full codec (like MP3), this is complex. 
    # For an adversarial attack, we use a "Spectral Envelope Following" approach.
    # We assume that if a frequency is LOUD, we can hide quiet noise underneath it.
    
    # '0.03' = masking ratio. 
    # Lower (0.01) = Less noise, better quality, weaker attack.
    # Higher (0.05) = More noise, worse quality, stronger attack.
    masking_threshold = magnitude * 0.03 
    
    return masking_threshold, np.angle(stft)

def apply_psychoacoustic_clipping(perturbation, audio, sr, window_size=2048, hop_size=512):
    """
    Clips the adversarial noise (perturbation) so it stays strictly 
    below the masking threshold of the original audio.
    """
    # Get the threshold from the ORIGINAL audio
    threshold, _ = compute_masking_threshold(audio, sr, window_size, hop_size)
    
    # Analyze the NOISE (perturbation)
    pert_stft = librosa.stft(perturbation, n_fft=window_size, hop_length=hop_size)
    pert_mag = np.abs(pert_stft)
    pert_phase = np.angle(pert_stft)
    
    # Clip the noise: 
    # If noise_volume > threshold_volume, clamp it down. 
    # If noise_volume < threshold_volume, leave it alone.
    clipped_mag = np.minimum(pert_mag, threshold)
    
    # Reconstruct the noise back to audio (Inverse STFT)
    clipped_stft = clipped_mag * np.exp(1j * pert_phase)
    clipped_perturbation = librosa.istft(clipped_stft, length=len(perturbation), hop_length=hop_size)
    
    return clipped_perturbation