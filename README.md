# Projet : Discrimination d'Images par Réseaux de Neurones (MLP vs CNN)

## Contexte du projet
Ce projet a été développé dans le cadre de ma troisième année de BUT Science des Données. Il s'agit de concevoir et d'évaluer un système de classification d'images réelles issues de la base de données Wang. Le jeu de données comprend 1000 images réparties équitablement en 10 classes sémantiques distinctes (Jungle, Plage, Monuments, Bus, Dinosaures, etc.). L'objectif de ce projet est de comparer deux paradigmes de vision par ordinateur : une approche traditionnelle basée sur des descripteurs pré-calculés et une approche Deep Learning "de bout en bout".

## Processus de réalisation
L'architecture technique et le développement de ce système de discrimination se sont déroulés en comparant deux stratégies distinctes.

### Approche Basée Modèle (Feature Engineering)
J'ai d'abord exploité des descripteurs FCTH (Fuzzy Color and Texture Histogram) pour caractériser les images. La préparation des données a impliqué l'étiquetage automatique, le partitionnement (Train, Validation, Test), la normalisation par centrage-réduction (StandardScaler) et l'encodage binaire des labels. J'ai ensuite construit un réseau Perceptron Multi-Couches (MLP) "Full-Connected" avec Keras et TensorFlow. J'ai expérimenté plusieurs architectures et méthodes de régularisation (Dropout, Batch Normalization) pour limiter le sur-apprentissage.

### Approche Deep Learning (Representation Learning)
Pour surmonter les limites des descripteurs statistiques, j'ai implémenté une seconde stratégie consistant à fournir les pixels bruts directement au réseau. Les images ont été redimensionnées en 128x128 pixels pour s'adapter aux contraintes de mémoire vive. J'ai développé et comparé deux architectures de Réseaux de Neurones Convolutifs (CNN) chargées d'apprendre automatiquement à extraire des caractéristiques spatiales hiérarchiques (des formes simples aux objets complexes).

## Évaluation et Diagnostic des Modèles
L'analyse des performances s'est appuyée sur l'étude des courbes d'apprentissage (Loss/Accuracy) et des matrices de confusion. J'ai simulé l'utilisation réelle du système en soumettant des images inconnues pour observer le taux de confiance des prédictions. Cette phase a permis d'identifier les limites de chaque modèle face à des contextes visuels trompeurs (par exemple, la confusion entre la classe "Plage" et "Montagne" due aux textures d'arrière-plan).

## Finalité
Le résultat est une étude comparative complète mettant en lumière la supériorité du Deep Learning pour les tâches de vision par ordinateur complexes. Le modèle CNN a atteint une précision de 71 %, surpassant largement l'approche par descripteurs qui plafonnait à environ 50 %. Ce projet démontre ma capacité à gérer l'intégralité d'un pipeline de Machine Learning, de la préparation de la donnée jusqu'à l'analyse critique des limites structurelles des modèles déployés.
