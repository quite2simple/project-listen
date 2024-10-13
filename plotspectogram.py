import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from tospectogram import audio_to_log_mel_spectrogram

SAMPLE_RATE = 22050
AUDIO_DURATION = 1  # in seconds
N_MELS = 128

t = np.linspace(0, AUDIO_DURATION, int(SAMPLE_RATE * AUDIO_DURATION), endpoint=False)
audio_fragment = np.random.normal(0, 1, int(SAMPLE_RATE * AUDIO_DURATION))
spectograms = audio_to_log_mel_spectrogram(audio_fragment, sample_rate=SAMPLE_RATE, n_mels=N_MELS)

for spectogram in spectograms:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectogram, sr=SAMPLE_RATE, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    # plt.tight_layout()
    plt.show()