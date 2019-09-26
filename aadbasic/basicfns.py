import numpy as np
import librosa
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def envextract(wavfilename, envtype='power'):
    if envtype is 'power':
        Fs, audio = wavfile.read(wavfilename)
        fs_down_1 = 8000
        fs_inter = 120
        fs_target = 20
        fl = 1
        fh = 8
        audio = audio.astype(float)
        audio_ds = librosa.core.resample(audio, Fs, fs_down_1)
        envelope_audio = np.absolute(audio_ds)**2
        envelope_audio = librosa.core.resample(audio, fs_down_1, fs_inter)
        envelope = butter_bandpass_filter(envelope_audio, fl, fh, fs_inter, 6)
        envelope = librosa.core.resample(envelope, fs_inter, fs_target)
        return [envelope]
