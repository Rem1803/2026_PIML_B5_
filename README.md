Projet PIML


Objectifs :
===
Le projet consiste en l'implémentation d'un workflow pour classer des images de cellules.
Le but est de différencier les cellules saines et les cellules infectées par la malaria parmis de multiples échantillons issus de nombreux patients.
Pour cela nous utiliserons un mélange d'algorithmes de traitement d'image, d'algorithme de machine learning et de visualisation de performance.



Présentation des datas :
===
(source, taille, contexte)

Les données sur lesquelles nous travaillons sont issues du "Malaria Cell Image Dataset", qui a été publié il y a 7 ans sur la plateforme Kaggle.
Lien : https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/data

Kaggle est une plateforme collaborative d'échanges, de publications et de cours sur les Data Sciences.
Le dataset pèse 350.95 MB et est composé de 27558 images RGB, séparées équitablement en deux dossiers : "Infected" et "Uninfected"

Chaque image consiste en une seule cellule, découpée d'une vue microscope plus large.
Les échantillons de cellules ont été récupérés sur des patients humains par frottis sanguins.
A première vue, les cellules du dossier infectées par la Malaria se différencient visuellement des cellules saines parla présence d'une ou plusieurs tâches violettes.

Pour des raisons d'espace de stockage Git et de rapidité d'exécution, nous n'utiliserons que 2000 éléments de notre dataset (1000 de chaque dossier)



Aperçu du workflow / intuition du processus
===
(les différentes étapes : input, output, process...)

Le workflow que nous avons implémenté se sépare en 3 grandes parties :
- pré-traitement des images et descripteurs
- modèle de réseau de neurones
- évaluation et interprétabilité

La première partie a pour but de transformer les images du dataset utilisé pour s'assurer qu'elles soient utilisable de façon optimisée par le réseau de neurones.
Les images sont ainsi redimensionnées pour être du même format, puis convertie en niveau d'intensité de couleur pour supprimer le paramètre "luminosité du microscope" ; elles sont enfin normalisées.
Des descripteurs alternatifs sont également calculés sur chaque image pour aider le réseau de neurones à différencier les cellules :
- la variance de l'image
- la proportion de violet de l'image
- la moyenne de la saturation de l'image
- le nombre de régions détecté après segmentation Watershed de l'image

La deuxième partie a pour objectif d’implémenter et de comparer différents modèles de perceptrons multicouches (MLP) afin d’identifier le plus performant. Nous avons commencé par développer un modèle simple puis nous avons introduit progressivement plusieurs améliorations telles que l’utilisation de la fonction d’activation ReLU, du dropout, de la Binary Cross-Entropy et du mini-batch gradient descent. 
Afin d’optimiser au mieux les performances du réseau, nous avons implémenté une fonction qui recherche les hyper paramètres optimaux, pour le learning rate et le nombre d’époques d'entraînement. 
Enfin, la sélection du meilleur modèle a été réalisée à partir des performances obtenues en accuracy et sur l’analyse des courbes d’apprentissage.

La troisième partie a pour but de valider la robustesse de notre meilleur modèle et d'analyser ses performances de manière critique, en gardant à l'esprit le contexte médical du projet. Pour nous assurer que le réseau généralise bien sur de nouvelles données et ne fait pas de surapprentissage, nous utilisons une méthode de validation croisée. Les prédictions du modèle sont ensuite décortiquées grâce à plusieurs métriques clés : l'Exactitude (Accuracy), la Précision, le Rappel (Recall) et le score F1.
Une attention toute particulière est accordée au Rappel et à la matrice de confusion globale. En effet, dans le cadre du diagnostic de la malaria, il est vital de minimiser les Faux Négatifs (déclarer saine une cellule infectée).



Instructions d'installation :
===
(qu'est-ce que ça nécessite ? Peut-il tourner sur Linux VM)

Pour faire fonctionner le projet, une version 3.12 (ou 3.13) de Python est conseillée

Notre workflow utilise plusieurs bibliothèques de fonctions Python. Il convient donc de vérifier que les bibliothèques ci-dessous sont bien installées sur la machine où il sera exécuté, et de les installer sinon à l'aide de la commande "pip install [nom du module]". Pour assurer la compatibilité entre ces modules, nous précisons également la version recommendée de chaque module à avoir sur l'environnement Python.
    numpy (version 1.26.4); skimage (version 0.24.0) ; matplotlib (version 3.9.0) ; scipy (version 1.13.1) ; os ; itertools ; sklearn ; random

Une fois cela fait, il faut télécharger l'intégralité des fichiers présents sur le Git et les placer dans un même dossier sur la machine.



Comment exécuter une démo :
===
(moins de 10min de temps d'éxécution sur Linux VM)

- Ouvrir le fichier modèle.ipynb dans un logiciel capable de gérer et exécuter un Notebook python.

- Exécuter la 1ère cellule de code pour importer les librairies nécessaires

- Exécuter les 2e et 3e cellules de code pour charger le dataset utilisé et afficher quelque exemples d'images de cellules saines et infectées par la malaria

- Exécuter
