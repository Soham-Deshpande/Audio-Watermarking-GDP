# Testing for Connor

import numpy as np 
import soundfile
import matplotlib.pyplot as plt
from functions import signalToNoise

# No exception handling whilst testing

# Default Audio
# Assumption running from Audio-Watermarking-GDP/
samplePath = "./Testing/Connor-Testing/samples/sample.flac"
sample, sampleRate = soundfile.read(samplePath)

plt.subplot(3, 1, 1) # This being 1 indexed makes me sad
plt.plot(sample)
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.xlim(left=0, right=len(sample))


# Gaussian Noise
sampleGaussianPath = "./Testing/Connor-Testing/samples/sampleGaussian.flac"
gaussianNoise = np.random.normal(size=len(sample)) * 0.05
sampleGaussian = sample + gaussianNoise
soundfile.write(sampleGaussianPath, sampleGaussian, sampleRate)

plt.subplot(3, 1, 2)
plt.plot(sampleGaussian)
plt.xlabel("Sample with Gaussian Noise")
plt.ylabel("Amplitude")
plt.xlim(left=0, right=len(sampleGaussian))

# High Frequency Signal
FREQUENCY = 10000 # Adjust to >20,000 to be inaudible
sampleHighFrequencyPath = "./Testing/Connor-Testing/samples/sampleHighFrequency.flac"
sampleTime = np.arange(0, len(sample)) / sampleRate
# 2pi, convert to radians
# Frequency, number of sine waves per second
# Sample time, time at point where sin is calculated
highFrequencySignal = 0.01 * np.sin(2 * np.pi * FREQUENCY * sampleTime)
sampleHighFrequency = sample + highFrequencySignal
soundfile.write(sampleHighFrequencyPath, sampleHighFrequency, sampleRate)

plt.subplot(3, 1, 3)
plt.plot(sampleHighFrequency)
plt.xlabel("Sample with High Frequency Noise")
plt.ylabel("Amplitude")
plt.xlim(left=0, right=len(sampleHighFrequency))
plt.show()

print(signalToNoise(sample, gaussianNoise * 0.02)) # Negligible noise - unknown actual?
print(signalToNoise(sampleGaussian, gaussianNoise))
print(signalToNoise(sampleHighFrequency, highFrequencySignal))

# TODO Spectrogram output