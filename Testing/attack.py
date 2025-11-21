import torch
import torch.nn as nn
import numpy as np
from .psychoacoustic import apply_psychoacoustic_clipping

class PsychoacousticPGD:
    def __init__(self, model, processor, device='cpu'):
        self.model = model
        self.processor = processor
        self.device = device

    def generate(self, audio_array, sr, epsilon=0.02, alpha=0.002, num_iter=40):
        """
        Generates an adversarial audio sample using PGD with Psychoacoustic Masking.
        
        Args:
            audio_array (np.array): The original audio signal.
            sr (int): Sample rate.
            epsilon (float): Maximum allowed noise level (L-infinity).
            alpha (float): Step size per iteration.
            num_iter (int): Number of attack iterations.
        """
        print(f"Starting PGD Attack ({num_iter} iterations)...")
        
        # 1. Prepare inputs
        inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        
        # Get the "Ground Truth" transcription (Pseudo-labeling)
        # We want to force the model to DEVIATE from this.
        with torch.no_grad():
            logits = self.model(input_values).logits
            target_ids = torch.argmax(logits, dim=-1)

        # Create a noise tensor (Delta)
        delta = torch.zeros_like(input_values, requires_grad=True).to(self.device)
        
        original_input = input_values.detach().clone()

        # Iterative Attack Loop
        for t in range(num_iter):
            # Combine audio + noise
            adversarial_input = original_input + delta
            
            # Forward pass
            outputs = self.model(adversarial_input)
            
            # Calculate Loss (CTC Loss against the ORIGINAL transcription)
            # We want to MAXIMIZE this loss to confuse the model.
            # Note: We use negative log likelihood or standard CTC loss
            log_probs = nn.functional.log_softmax(outputs.logits, dim=-1).transpose(0, 1)
            
            # CTC Loss requires lengths
            input_lengths = torch.full(size=(1,), fill_value=log_probs.size(0), dtype=torch.long).to(self.device)
            target_lengths = torch.full(size=(1,), fill_value=target_ids.size(1), dtype=torch.long).to(self.device)
            
            # Calculate standard CTC loss
            loss = nn.functional.ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
            
            # We want to INCREASE the error, so we do Gradient ASCENT (positive update)
            loss.backward()
            
            # Update noise
            grad = delta.grad.detach()
            delta.data = delta.data + alpha * grad.sign()
            
            # Clip to epsilon (standard PGD constraint)
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            
            # Zero gradients for next step
            delta.grad.zero_()
            
            # --- PSYCHOACOUSTIC STEP ---
            # enforce the masking at the end
            if t % 5 == 0 or t == num_iter - 1:
                pert_np = delta.data.cpu().numpy().flatten()
                audio_np = original_input.cpu().numpy().flatten()
                
                # Apply the masking from our helper file
                refined_pert = apply_psychoacoustic_clipping(pert_np, audio_np, sr)
                
                # Update the delta tensor with the refined noise
                delta.data = torch.tensor(refined_pert).float().reshape(delta.shape).to(self.device)

        # 5. Final Output
        adv_audio = (original_input + delta).detach().cpu().numpy().flatten()
        return adv_audio