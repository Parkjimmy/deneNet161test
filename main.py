import torch
from preprocessor import SEMGPreprocessor
from dataset import load_raw_dataset
from models import get_baseline_2dcnn, get_baseline_resnet18, get_densenet161
from train import train_and_eval_model
from utils import plot_confusion_matrix

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")

    data_path = "palm-sEMG-doorknob-filtered-main/data"
    trial_signals, trial_labels, _ = load_raw_dataset(data_path)
    print(f"총 로드된 시행(Trial) 수: {len(trial_signals)}개")

    preprocessor = SEMGPreprocessor(fs=1000, window_size=300, overlap=150, num_scales=32)

    models_to_test = [
        ("Baseline 2D CNN", get_baseline_2dcnn, "cm_cnn.png"),
        ("ResNet18", get_baseline_resnet18, "cm_resnet18.png"),
        ("DenseNet161", get_densenet161, "cm_densenet161.png")
    ]

    results = []
    for name, fn, save_img in models_to_test:
        res = train_and_eval_model(fn, name, trial_signals, trial_labels, preprocessor, device, epochs=30)
        plot_confusion_matrix(res['trues'], res['preds'], model_name=name, save_path=save_img)
        results.append(res)

    print("\n" + "="*60)
    print(" [최종 모델 성능 비교 표 요약] ")
    print("="*60)
    print(f"{'Model':<18} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-score':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['model_name']:<18} | {r['accuracy']*100:6.2f}%  | {r['precision']*100:7.2f}%  | {r['recall']*100:6.2f}%  | {r['f1_score']*100:6.2f}%")
    print("="*60)

if __name__ == '__main__':
    main()