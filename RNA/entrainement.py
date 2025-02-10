import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Définir les dimensions des images
img_height, img_width = 64, 64

# Créer un générateur d'images pour l'entraînement
train_datagen = ImageDataGenerator(rescale=1./255)

# Charger les images d'entraînement à partir du dossier 'img'
train_generator = train_datagen.flow_from_directory(
    '/img',
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

# Définir le modèle de réseau de neurones
model = Sequential()
model.add(Dense(5000, input_shape=(img_height, img_width, 3), activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(train_generator, epochs=10)

# Sauvegarder le modèle entraîné
model.save('RNA/model.h5')

print("Modèle entraîné et sauvegardé avec succès.")
