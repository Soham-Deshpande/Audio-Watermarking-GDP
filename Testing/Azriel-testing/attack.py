import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torchaudio.transforms as T
from psychoacoustic import apply_psychoacoustic_clipping

class PsychoacousticPGD:
    def __init__(self, model, processor, device='cpu'):
        self.model = model
        self.processor = processor
        self.device = device
        
        # --- ROBUSTNESS TOOLS  ---
        self.resample_down = T.Resample(orig_freq=16000, new_freq=8000).to(device)
        self.resample_up = T.Resample(orig_freq=8000, new_freq=16000).to(device)

    def generate(self, audio_array, sr, epsilon=0.02, alpha=0.002, num_iter=50):
        """
        Generates an adversarial audio sample using Robust PGD.
        The noise is trained to survive compression and static.
        """
        print(f"Starting ROBUST PGD Attack ({num_iter} iterations)...")
        
        # Prepare inputs
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        
        # Get the "Ground Truth" to deviate from
        with torch.no_grad():
            logits = self.model(input_values).logits
            target_ids = torch.argmax(logits, dim=-1)

        # Initialize noise (delta)
        delta = torch.zeros_like(input_values, requires_grad=True).to(self.device)
        original_input = input_values.detach().clone()

        for t in range(num_iter):
            # Combine audio + noise
            adversarial_input = original_input + delta
            
            # --- THE UPGRADE: Simulation Layer (EOT) ---
            # We don't just feed the clean audio to the model.
            # We randomly "mess it up" so the noise learns to survive.
            
            rand_val = torch.rand(1).item()
            
            if rand_val < 0.33:
                # Option A: Add Static Noise (Simulate background noise)
                noise = torch.randn_like(adversarial_input) * 0.005
                transformed_input = adversarial_input + noise
                
            elif rand_val < 0.66:
                 # Option B: Resample (Simulate 8kHz Phone Line/Compression)
                downsampled = self.resample_down(adversarial_input)
                transformed_input = self.resample_up(downsampled)
                
                # Safety check: Resampling sometimes changes length slightly due to rounding
                if transformed_input.shape != adversarial_input.shape:
                     transformed_input = adversarial_input
            else:
                # Option C: Clean (Control group)
                transformed_input = adversarial_input

            # Forward pass (using the MESSY input)
            outputs = self.model(transformed_input)
            
            # Calculate Loss (CTC)
            # We want to MAXIMIZE the loss (gradient ascent) to confuse the model
            log_probs = nn.functional.log_softmax(outputs.logits, dim=-1).transpose(0, 1)
            
            input_lengths = torch.full(size=(1,), fill_value=log_probs.size(0), dtype=torch.long).to(self.device)
            target_lengths = torch.full(size=(1,), fill_value=target_ids.size(1), dtype=torch.long).to(self.device)
            
            loss = nn.functional.ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
            
            # Update Noise (Gradient Ascent)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = delta.data + alpha * grad.sign()
            
            # Clip to epsilon constraint
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            # Reset gradients
            delta.grad.zero_()
            
            # Psychoacoustic Cleanup (Keep it invisible)
            if t % 5 == 0 or t == num_iter - 1:
                pert_np = delta.data.cpu().numpy().flatten()
                audio_np = original_input.cpu().numpy().flatten()
                
                refined_pert = apply_psychoacoustic_clipping(pert_np, audio_np, sr)
                
                delta.data = torch.tensor(refined_pert).float().reshape(delta.shape).to(self.device)

        # Final Output
        adv_audio = (original_input + delta).detach().cpu().numpy().flatten()
        return adv_audio
