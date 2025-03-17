import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import matplotlib.pyplot as plt

def copy_random_images(src_dir, dst_dir, num_images):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    images = os.listdir(src_dir)
    selected_images = random.sample(images, num_images)
    for img in selected_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

def prepare_test_data():
    validation_data_dir = 'data/validation/'
    test_data_dir = 'data/test/'
    result_dir = 'data/result/'
    
    # Kiểm tra và tạo thư mục test nếu không tồn tại
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    
    # Xóa tất cả các tệp hoặc thư mục con trong thư mục test nếu có
    for filename in os.listdir(test_data_dir):
        file_path = os.path.join(test_data_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    
    # Xóa thư mục result nếu tồn tại
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    
    # Tạo thư mục result và các thư mục con dog và cat
    os.makedirs(os.path.join(result_dir, 'dog'))
    os.makedirs(os.path.join(result_dir, 'cat'))
    
    # Sao chép ngẫu nhiên 50 ảnh chó và 50 ảnh mèo vào thư mục test
    copy_random_images(os.path.join(validation_data_dir, 'dog'), test_data_dir, 50)
    copy_random_images(os.path.join(validation_data_dir, 'cat'), test_data_dir, 50)
    
    return test_data_dir, result_dir

def predict_model(test_data_dir, result_dir):
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,  # Không có nhãn
        shuffle=False,
        classes=['.']  # Không yêu cầu các thư mục con
    )
    
    model = load_model('models/dog_cat_model.h5')
    predictions = model.predict(test_generator)
    predicted_classes = np.where(predictions > 0.5, 1, 0)
    
    # In ra tỉ lệ nhận diện
    num_dogs = np.sum(predicted_classes == 1)
    num_cats = np.sum(predicted_classes == 0)
    total = len(predicted_classes)
    print(f"Tỉ lệ nhận diện chó: {num_dogs / total * 100:.2f}%")
    print(f"Tỉ lệ nhận diện mèo: {num_cats / total * 100:.2f}%")
    
    # Vẽ biểu đồ tỉ lệ nhận diện
    labels = ['Dogs', 'Cats']
    sizes = [num_dogs, num_cats]
    colors = ['#ff9999','#66b3ff']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Tỉ lệ nhận diện chó và mèo')
    plt.show()
    
    # Di chuyển các ảnh dự đoán vào các thư mục result/dog và result/cat tương ứng
    filenames = test_generator.filenames
    for i, filename in enumerate(filenames):
        src_path = os.path.join(test_data_dir, filename)
        if predicted_classes[i] == 1:
            dst_path = os.path.join(result_dir, 'dog', os.path.basename(filename))
        else:
            dst_path = os.path.join(result_dir, 'cat', os.path.basename(filename))
        shutil.move(src_path, dst_path)
    
    print(predicted_classes)

if __name__ == "__main__":
    test_data_dir, result_dir = prepare_test_data()
    predict_model(test_data_dir, result_dir)