import torch
from utils import load_model, load_sample_data

def calculate_map10(predictions, labels):
    """Dummy MAP@10 calculation."""
    # 실제 MAP@10 계산 로직을 여기에 작성
    return 0.85

def calculate_uncertainty(predictions):
    """Calculate uncertainty (dummy implementation)."""
    return predictions.var(dim=1).item()

def main():
    model_path = "models/AL_model.pt"
    data_path = "data/sample.csv"
    input_dim = 10  # 샘플 데이터의 피처 수

    # 모델 및 데이터 로드
    model = load_model(model_path, input_dim)
    data = load_sample_data(data_path)

    # 예측 수행
    with torch.no_grad():
        predictions = model(data)
        probabilities = torch.softmax(predictions, dim=1)

    # MAP@10 및 불확실성 계산
    map10 = calculate_map10(probabilities, labels=None)  # Labels 필요 시 추가
    uncertainty = calculate_uncertainty(probabilities)

    print(f"MAP@10: {map10}")
    print(f"Uncertainty: {uncertainty}")
    print(f"Probabilities: {probabilities}")

if __name__ == "__main__":
    main()
