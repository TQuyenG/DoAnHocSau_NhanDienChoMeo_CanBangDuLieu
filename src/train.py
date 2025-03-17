from src.data_preparation import prepare_validation_split
from src.model import create_simple_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model():
    prepare_validation_split(val_size=0.2)

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    validation_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        'data/validation/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Tạo mô hình
    model = create_simple_cnn()
    
    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Huấn luyện mô hình
    model.fit(train_generator, validation_data=validation_generator, epochs=10)

    # Lưu mô hình
    model.save('models/dog_cat_model.h5')

if __name__ == "__main__":
    train_model()