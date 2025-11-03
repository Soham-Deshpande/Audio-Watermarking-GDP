import numpy as np 
from audiomentations import Compose, AddGaussianNoise, TimeStretch

audio_folder = '../src/dataset/LibriSpeech/train-clean-100'

audio_clip = audio_folder + '\19\198\19-198-0001.flac'
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
])

augmented_sample = augment(samples=audio_clip,sample_rate=16000)
