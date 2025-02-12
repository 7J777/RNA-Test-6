# RNA-test-6

Ce projet utilise un réseau de neurones artificiels pour l'entraînement d'un modèle de classification d'images utilisant Keras et TensorFlow.

## Structure du projet

- `.github/workflows/` : Contient les fichiers de configuration pour GitHub Actions.
- `RNA/` : Contient le script d'entraînement du modèle.
- `img/` : Contient les images organisées en sous-répertoires représentant les classes.

## Exécution du workflow CI

Le workflow CI est configuré pour s'exécuter à chaque push ou pull request sur la branche `main`. Il effectue les étapes suivantes :

1. Vérifie le dépôt.
2. Configure Python 3.8.
3. Installe les dépendances nécessaires (`keras`, `pillow`, `numpy`, `tensorflow`, `scipy`).
4. Exécute le script d'entraînement du modèle (`RNA/entrainement.py`).
5. Télécharge le modèle entraîné en tant qu'artifact (`RNA/model.h5`).
6. Ajoute le fichier `model.h5` au dépôt dans le dossier `RNA`.

### Accès au modèle entraîné

Pour accéder au fichier `model.h5` après l'entraînement du modèle :

1. Clonez le dépôt pour obtenir la dernière version du modèle.
2. Le fichier `model.h5` se trouve dans le dossier `RNA`.

## Utilisation du modèle entraîné

Pour utiliser le modèle `model.h5` dans votre propre script Python, suivez les étapes ci-dessous :

*(A venir n'a pas encore été définis)*
