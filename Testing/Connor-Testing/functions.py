# Useful functions - just for keeping test code more tidy

import numpy as np

# https://en.wikipedia.org/wiki/Signal-to-noise_ratio
# Signal to noise ratio in decibels
def signalToNoise(signal, noise) -> float:
  signal = np.mean(signal ** 2)
  noise = np.mean(noise ** 2)
  return 10 * np.log10(signal / noise)
