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

### Accès au modèle entraîné

Pour accéder au fichier `model.h5` après l'entraînement du modèle :

1. Allez dans l'onglet "Actions" de votre dépôt GitHub.
2. Sélectionnez le workflow CI.
3. Choisissez la dernière exécution réussie du workflow.
4. Téléchargez l'artifact nommé `trained-model`.
