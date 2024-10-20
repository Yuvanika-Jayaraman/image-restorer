import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
def load_image(path):
    return cv2.imread(path)
def add_noise(image):
    noise = np.random.normal(loc=0, scale=0.1, size=image.shape).astype('float32')
    noisy = cv2.add(image, noise)
    return np.clip(noisy, 0.0, 1.0)
def build_denoising_model(input_shape):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), padding='same')
    ])
    return model
def train_model(model, noisy, clean, epochs=10, batch_size=32):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(noisy, clean, epochs=epochs, batch_size=batch_size, validation_split=0.0)
def save_image(image, path):
    cv2.imwrite(path, (image * 255).astype('uint8'))
def process_single_image(image_path, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)
    og = load_image(image_path)
    og= cv2.resize(og, (256, 256))
    og= og.astype('float32') / 255.0
    noisy = add_noise(og)
    denoised = model.predict(noisy[np.newaxis, ...])[0]
    save_image(og, os.path.join(output_folder, 'original_image.png'))
    save_image(noisy, os.path.join(output_folder, 'noisy_image.png'))
    save_image(denoised, os.path.join(output_folder, 'denoised_image.png'))
    print(f"Images saved in folder: {output_folder}")

def main():
    image_path = input("Enter the path to the image file: ")
    output_folder = input("Enter the output folder path: ")
    print(f"Using image file: '{image_path}'")
    og = load_image(image_path)
    og = cv2.resize(og, (256, 256))
    og = og.astype('float32') / 255.0
    noisy = add_noise(og)
    model = build_denoising_model((256, 256, 3))
    train_model(model, np.array([noisy]), np.array([og]))
    process_single_image(image_path, output_folder, model)

if __name__ == "__main__":
    main()

