# Projet de Classification de Pneumonie

Ce projet vise à classifier les images de radiographies thoraciques pour détecter les cas de pneumonie à l'aide d'un réseau de neurones convolutif (CNN).

## Structure du Projet



- **data/chestX_ray** : Ce répertoire contient le dataset de radiographies thoraciques, organisé en sous-répertoires pour l'entraînement, la validation et les tests.
    - **train** : Contient les images d'entraînement.
  
    - **test** : Contient les images de test.
  
    - **val** : Contient les images de validation.


- **cnn.ipynb** : Notebook Jupyter contenant le code pour construire, entraîner et évaluer le modèle de réseau de neurones convolutif.



- **requirements.txt** : Fichier listant les packages nécessaires à l'exécution du projet. Utilisez la commande suivante pour installer les dépendances :
  ```bash
  pip install -r requirements.txt
    ```


- **metrics.md**: Fichier Markdown définissant les différentes métriques utilisées en deep learning pour évaluer les performances du modèle.


