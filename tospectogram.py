import numpy as np
import librosa


def audio_to_log_mel_spectrogram(audio_fragment: np.ndarray, sample_rate=22050, n_mels=128, hop_length=512, n_fft=2048):
    """
    Convert audio fragment to log mel spectrogram
    """

    # if the fragment is 1D, convert it to 2D

    if audio_fragment.ndim == 1:
        audio_fragment = audio_fragment.reshape(1, -1)
    
    # return value is a list of spectograms
    res = []

    # loop over channels, converting each into a spectogram

    for channel in audio_fragment:

        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_fragment[0],
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft
        )

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        res.append(log_mel_spectrogram)

    return res


if __name__ == "__main__":

    SAMPLE_RATE = 22050
    AUDIO_DURATION = 1  # in seconds
    N_MELS = 128

    audio_fragment = np.sin(2 * np.pi * 440 * np.linspace(0, AUDIO_DURATION, SAMPLE_RATE * AUDIO_DURATION))

    spectograms = audio_to_log_mel_spectrogram(audio_fragment, sample_rate=SAMPLE_RATE, n_mels=N_MELS)

    print(spectograms)