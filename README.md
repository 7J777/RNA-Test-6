# RNA-test-6

Ce projet utilise un réseau de neurones artificiels pour l'entraînement d'un modèle de classification d'images utilisant `Keras` et `TensorFlow`.

## Structure du projet

- `.github/workflows/` : Contient les fichiers de configuration pour GitHub Actions.
- `RNA/` : Contient le script d'entraînement du modèle.
- `img/` : Contient les images organisées en sous-répertoires représentant les classes.

## Exécution du workflow CI

Le workflow `CI` (`main.yml`) est configuré pour s'exécuter à chaque push ou pull request sur la branche `main`. Il effectue les étapes suivantes :

1. Vérifie le dépôt.
2. Configure `Python 3.8`.
3. Installe les dépendances nécessaires (`keras`, `pillow`, `numpy`, `tensorflow`, `scipy`).
4. Exécute le script d'entraînement du modèle (`RNA/entrainement.py`).
5. Télécharge le modèle entraîné en tant qu'artifact (`RNA/model.h5`).

### Accès au modèle entraîné

Pour accéder au fichier `model.h5` après l'entraînement du modèle :

1. Allez dans l'onglet `Action` cherchez la dernière exécution.
2. Télécharger `model.h5` en tant qu'artifact.

## Utilisation du modèle entraîné

Pour utiliser le modèle `model.h5` suivez les étapes ci-dessous :

*(A venir n'a pas encore été définis)*

##   
#### Modification à venir

1. Ajout dans le workflow d'une automatisation qui télécharge le modèle `model.h5` dans le répertoire `/RNA`
2. intégration d'un script capable d'exécuter `model.h5`
3. Ajout d'une application ou d'un site web qui utilise ce script
4. Ajout d'imlage dans le dossier `img`

