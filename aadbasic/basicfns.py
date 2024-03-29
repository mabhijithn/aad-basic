import numpy as np
import librosa
import resampy
import pickle
import scipy.io.wavfile as wavfile
from scipy.signal import butter, lfilter, filtfilt

def save_obj(tardir, name, obj ):
    with open(tardir + name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(tardir, name):
    with open(tardir + name + '.pkl', 'rb+') as f:
        return pickle.load(f)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)    
    y = filtfilt(b, a, data)
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
        audio_ds = resampy.resample(audio, Fs, fs_down_1)
        envelope_audio = np.absolute(audio_ds)**2        
        envelope_audio = resampy.resample(envelope_audio, fs_down_1, fs_inter)
        envelope = butter_bandpass_filter(envelope_audio, fl, fh, fs_inter, 6)        
        envelope = resampy.resample(envelope, fs_inter, fs_target)
        envdict = {
            "envelope" : envelope,
            "Fs" : fs_target,
            "powerlaw": True,
            "subbands": False
        }
        return envdict
