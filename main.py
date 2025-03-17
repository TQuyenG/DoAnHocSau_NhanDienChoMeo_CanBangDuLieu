from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    # Huấn luyện mô hình
    train_model()
    
    # Đánh giá mô hình
    evaluate_model()