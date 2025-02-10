import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# Afficher le contenu du répertoire
data_dir = '/img'
print(f"Contenu de {data_dir} :")
for root, dirs, files in os.walk(data_dir):
    for name in files:
        print(os.path.join(root, name))

# Configuration du générateur de données
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,  # Utilisez None si vous n'avez pas de classes
    shuffle=True
)

# Modèle avec l'architecture 5000 -> 500 -> 50 -> 1
model = Sequential([
    Flatten(input_shape=(150, 150, 3)),
    Dense(5000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(train_generator, epochs=10)
