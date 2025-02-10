import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten

print("Début du script")

# Afficher le contenu du répertoire
data_dir = './img'  # Utilisez un chemin relatif pour le répertoire img
print(f"Contenu de {data_dir} :")
if not os.path.exists(data_dir):
    raise ValueError(f"Le répertoire {data_dir} n'existe pas.")
for root, dirs, files in os.walk(data_dir):
    for name in files:
        print(os.path.join(root, name))

# Vérifier la structure du répertoire
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
if len(classes) < 2:
    raise ValueError(f"Le répertoire {data_dir} doit contenir au moins deux sous-répertoires représentant les classes. Classes trouvées : {classes}")

print(f"Classes trouvées : {classes}")

# Vérifier s'il y a des images dans les sous-répertoires
images_found = False
for class_dir in classes:
    class_path = os.path.join(data_dir, class_dir)
    if any(file.endswith(('png', 'jpg', 'jpeg')) for file in os.listdir(class_path)):
        images_found = True
        break

if not images_found:
    raise ValueError(f"Aucune image trouvée dans les sous-répertoires de {data_dir}.")

print("Images trouvées")

# Configuration du générateur de données
datagen = ImageDataGenerator(rescale=1./255)
try:
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',  # Utilisez 'binary' ou 'categorical' selon votre cas d'utilisation
        shuffle=True
    )
    print("Générateur de données créé")
except Exception as e:
    print(f"Erreur lors de la création du générateur de données: {e}")
    raise

# Modèle avec l'architecture 5000 -> 500 -> 50 -> 1
try:
    model = Sequential([
        Flatten(input_shape=(150, 150, 3)),
        Dense(5000, activation='relu'),
        Dense(500, activation='relu'),
        Dense(50, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Modèle créé et compilé")

    # Entraînement du modèle
    model.fit(train_generator, epochs=10)
    print("Entraînement du modèle terminé")
except Exception as e:
    print(f"Erreur lors de l'entraînement du modèle: {e}")
    raise
