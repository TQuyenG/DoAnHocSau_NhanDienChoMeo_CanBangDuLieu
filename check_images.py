from PIL import Image
import os

def check_images(folder):
    """Kiểm tra tất cả các tệp trong thư mục để xác định xem có tệp hình ảnh nào bị hỏng không."""
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = Image.open(os.path.join(folder, filename))
                img.verify()  # Kiểm tra tệp hình ảnh
            except Exception as e:
                print(f"Lỗi với tệp {filename}: {e}")

if __name__ == "__main__":
    print("Kiểm tra hình ảnh trong thư mục 'data/train/dog':")
    check_images('data/train/dog')
    
    print("\nKiểm tra hình ảnh trong thư mục 'data/train/cat':")
    check_images('data/train/cat')