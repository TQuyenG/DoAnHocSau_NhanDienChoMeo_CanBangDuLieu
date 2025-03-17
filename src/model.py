from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

def create_simple_cnn(input_shape=(150, 150, 3), num_classes=1):
    """
    Tạo một mô hình CNN đơn giản cho bài toán phân loại chó mèo.
    
    Args:
        input_shape (tuple): Kích thước đầu vào (mặc định: 150x150x3)
        num_classes (int): Số lớp đầu ra (mặc định: 1 cho nhị phân)
        
    Returns:
        model: Mô hình Keras
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    return model

def create_vgg16_model(input_shape=(150, 150, 3), num_classes=1):
    """
    Tạo một mô hình dựa trên kiến trúc VGG16 với transfer learning.
    
    Args:
        input_shape (tuple): Kích thước đầu vào
        num_classes (int): Số lớp đầu ra
        
    Returns:
        model: Mô hình Keras
    """
    # Tải mô hình VGG16 đã được huấn luyện trên tập dữ liệu ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Đóng băng các lớp của mô hình cơ sở
    for layer in base_model.layers:
        layer.trainable = False
    
    # Thêm các lớp tùy chỉnh trên đầu mô hình cơ sở
    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')
    ])
    
    return model

def create_resnet50_model(input_shape=(150, 150, 3), num_classes=1):
    """
    Tạo một mô hình dựa trên kiến trúc ResNet50 với transfer learning.
    
    Args:
        input_shape (tuple): Kích thước đầu vào
        num_classes (int): Số lớp đầu ra
        
    Returns:
        model: Mô hình Keras
    """
    # Tải mô hình ResNet50 đã được huấn luyện trên tập dữ liệu ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)