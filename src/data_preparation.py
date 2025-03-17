import os
import random
import shutil
from sklearn.model_selection import train_test_split

def create_validation_directory():
    """Tạo thư mục validation nếu chưa tồn tại"""
    directories = [
        'data/validation/dog', 
        'data/validation/cat'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Đã tạo thư mục: {directory}")

def prepare_validation_split(val_size=0.2, random_state=42):
    """
    Chia một phần dữ liệu từ tập train để tạo tập validation.
    
    Args:
        val_size (float): Tỉ lệ dữ liệu dành cho kiểm định
        random_state (int): Seed cho việc tạo số ngẫu nhiên
    """
    # Tạo thư mục validation
    create_validation_directory()
    
    # Đường dẫn thư mục
    dog_dir = 'data/train/Dog'
    cat_dir = 'data/train/Cat'
    
    # Lấy danh sách các file ảnh
    dog_files = [f for f in os.listdir(dog_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    cat_files = [f for f in os.listdir(cat_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Phân chia dữ liệu cho tập validation
    dog_train, dog_val = train_test_split(dog_files, test_size=val_size, random_state=random_state)
    cat_train, cat_val = train_test_split(cat_files, test_size=val_size, random_state=random_state)
    
    # Di chuyển các file chó vào thư mục validation
    for file in dog_val:
        shutil.move(
            os.path.join(dog_dir, file),
            os.path.join('data/validation/dog', file)
        )
    
    # Di chuyển các file mèo vào thư mục validation
    for file in cat_val:
        shutil.move(
            os.path.join(cat_dir, file),
            os.path.join('data/validation/cat', file)
        )
    
    print(f"Số lượng ảnh chó trong tập huấn luyện: {len(dog_train)}")
    print(f"Số lượng ảnh mèo trong tập huấn luyện: {len(cat_train)}")
    print(f"Số lượng ảnh chó trong tập kiểm định: {len(dog_val)}")
    print(f"Số lượng ảnh mèo trong tập kiểm định: {len(cat_val)}")

def rename_folders():
    """Đổi tên thư mục từ 'Cat' -> 'cat' và 'Dog' -> 'dog'"""
    try:
        # Kiểm tra xem thư mục 'Cat' và 'Dog' có tồn tại không
        if os.path.exists('data/train/Cat'):
            # Tạo thư mục tạm để tránh vấn đề về phân biệt chữ hoa/thường trên Windows
            os.rename('data/train/Cat', 'data/train/cat_temp')
            os.rename('data/train/cat_temp', 'data/train/cat')
            print("Đã đổi tên thư mục 'Cat' -> 'cat'")
            
        if os.path.exists('data/train/Dog'):
            os.rename('data/train/Dog', 'data/train/dog_temp')
            os.rename('data/train/dog_temp', 'data/train/dog')
            print("Đã đổi tên thư mục 'Dog' -> 'dog'")
            
    except Exception as e:
        print(f"Lỗi khi đổi tên thư mục: {e}")

if __name__ == "__main__":
    # Đổi tên thư mục trước
    rename_folders()
    
    # Sau đó chia dữ liệu validation
    prepare_validation_split(val_size=0.2)