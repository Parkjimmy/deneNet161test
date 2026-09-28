import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def read_clean_csv(fpath):
    """
    CSV 파일 내 'Comp Ch 1', 'Comp Ch 2' 등 텍스트 헤더가 포함되어 있어도
    순수 수치 데이터만 자동으로 파싱하여 추출합니다.
    """
    df = pd.read_csv(fpath, header=None)
    # 숫자로 변환 불가능한 텍스트 헤더/문자열을 NaN으로 처리
    df_num = df.apply(pd.to_numeric, errors='coerce')
    # 헤더 행 등 모든 값이 NaN인 행과 열 제거
    df_num = df_num.dropna(how='all', axis=0).dropna(how='all', axis=1)
    # 혹시 남은 결측치가 있다면 0으로 대체
    df_num = df_num.fillna(0.0)
    return df_num.values.astype(np.float32)

def load_raw_dataset(data_root="palm-sEMG-doorknob-filtered-main/data"):
    classes = ['A', 'B', 'C', 'D', 'E']
    label_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    trial_signals = []
    trial_labels = []
    trial_filenames = []

    for cls_name in classes:
        folder_path = os.path.join(data_root, cls_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"경로를 찾을 수 없습니다: {folder_path}")

        files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
        print(f"클래스 [{cls_name}]: {len(files)}개 파일 로드 완료")

        for fpath in files:
            ext = os.path.splitext(fpath)[1].lower()
            
            if ext == '.csv':
                arr = read_clean_csv(fpath)
            elif ext == '.npy':
                arr = np.load(fpath)
            elif ext in ['.txt', '.dat']:
                arr = np.loadtxt(fpath)
            else:
                continue
            
            # (샘플 수, 채널 수) 형태인 경우 (채널 수, 샘플 수)로 Transpose 처리
            if arr.shape[0] > arr.shape[1]:
                arr = arr.T
                
            trial_signals.append(arr.astype(np.float32))
            trial_labels.append(label_map[cls_name])
            trial_filenames.append(os.path.basename(fpath))

    return trial_signals, np.array(trial_labels), trial_filenames


class SEMGDataset(Dataset):
    def __init__(self, trial_signals, trial_labels, preprocessor):
        self.samples = []
        self.labels = []

        for signal, label in zip(trial_signals, trial_labels):
            filtered = preprocessor.filter_signal(signal)
            segments = preprocessor.segment_and_normalize(filtered)
            for seg in segments:
                tensor_img = preprocessor.transform_to_cwt_tensor(seg)
                self.samples.append(tensor_img)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)