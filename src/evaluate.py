from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model():
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    
    validation_generator = validation_datagen.flow_from_directory(
        'data/validation/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    model = load_model('models/dog_cat_model.h5')
    
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

if __name__ == "__main__":
    evaluate_model()