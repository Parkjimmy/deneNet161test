import numpy as np
from scipy.signal import butter, iirnotch, filtfilt
import pywt  # PyWavelets 라이브러리

class SEMGPreprocessor:
    def __init__(self, fs=1000, window_size=300, overlap=150, num_scales=32):
        self.fs = fs
        self.window_size = window_size
        self.overlap = overlap
        self.num_scales = num_scales

    def filter_signal(self, raw_signal):
        # 1. 60Hz 노치 필터
        bn, an = iirnotch(60.0, 30.0, self.fs)
        filtered = filtfilt(bn, an, raw_signal, axis=-1)
        
        # 2. 20~495Hz 대역통과 필터
        nyq = self.fs / 2.0
        b, a = butter(4, [20.0 / nyq, 495.0 / nyq], btype='band')
        filtered = filtfilt(b, a, filtered, axis=-1)
        return filtered

    def segment_and_normalize(self, signal):
        num_samples = signal.shape[1]
        step = self.window_size - self.overlap
        segments = []

        for start in range(0, num_samples - self.window_size + 1, step):
            seg = signal[:, start:start + self.window_size]
            seg_min = seg.min(axis=1, keepdims=True)
            seg_max = seg.max(axis=1, keepdims=True)
            seg_norm = (seg - seg_min) / (seg_max - seg_min + 1e-8)
            segments.append(seg_norm)

        return np.array(segments)

    def transform_to_cwt_tensor(self, segment):
        widths = np.arange(1, self.num_scales + 1)
        
        # pywt.cwt는 (coefs, freqs) 튜플을 반환하므로 계수(coefs)만 추출
        cwt_ch1, _ = pywt.cwt(segment[0], widths, 'morl')
        cwt_ch2, _ = pywt.cwt(segment[1], widths, 'morl')
        
        cwt_ch1 = np.abs(cwt_ch1)
        cwt_ch2 = np.abs(cwt_ch2)
        cwt_ch3 = (cwt_ch1 + cwt_ch2) / 2.0

        tensor_3d = np.stack([cwt_ch1, cwt_ch2, cwt_ch3], axis=0)
        return tensor_3d.astype(np.float32)