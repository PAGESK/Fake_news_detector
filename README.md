# Fake News Detector

Ce projet a pour objectif de développer un modèle de détection de fausses nouvelles (fake news) basé sur des techniques de traitement du langage naturel et de machine learning.

## Description Technique

Ce projet implémente un pipeline de classification textuelle pour déterminer la véracité d'articles de presse. Le processus implique plusieurs étapes clés :

1.  **Acquisition des Données :** Utilisation de jeux de données étiquetés (`WELFake_Dataset.csv`, `df_fake_true.csv`) contenant des articles et leur label de véracité (vrai/faux).
2.  **Prétraitement du Texte :** Application d'une série de transformations aux textes bruts pour les rendre exploitables par les modèles :
    *   Conversion en minuscules.
    *   Remplacement des nombres par un token générique (`_num_`) ou suppression.
    *   Suppression des caractères spéciaux et de la ponctuation.
    *   Tokenization : Division du texte en mots ou unités significatives.
    *   Lemmatisation : Réduction des mots à leur forme de base (lemme) pour regrouper les variations morphologiques.
    *   Suppression des mots vides (stop words) : Retrait des mots fréquents mais peu informatifs (liste personnalisée incluant des termes spécifiques au domaine des actualités).
    Ces étapes sont encapsulées dans un `FunctionTransformer` pour une intégration facile dans un pipeline scikit-learn.
3.  **Vectorisation des Textes :** Conversion des textes prétraités en vecteurs numériques. La méthode **TF-IDF (Term Frequency-Inverse Document Frequency)** est utilisée pour pondérer l'importance des termes dans chaque document par rapport à l'ensemble du corpus. Le `TfidfVectorizer` est configuré pour extraire un nombre défini de features (`max_features`) et peut considérer des unigrammes et/ou des bigrammes (`ngram_range`).
4.  **Modélisation et Entraînement :** Le projet explore l'utilisation de différents algorithmes de classification performants pour les tâches de classification de texte :
    *   **Multinomial Naive Bayes :** Un modèle probabiliste simple mais efficace, particulièrement adapté aux données discrètes comme les comptes de mots ou les fréquences TF-IDF.
    *   **Logistic Regression :** Un modèle linéaire qui apprend les poids des features pour classer les documents.
    *   **LinearSVC (Support Vector Classifier) :** Une implémentation linéaire de la machine à vecteurs de support, connue pour sa performance sur des données de haute dimensionnalité comme les vecteurs TF-IDF.
    Les modèles sont entraînés sur un ensemble d'entraînement (`X_train`, `y_train`) issu d'une division stratifiée du jeu de données original.
5.  **Optimisation des Hyperparamètres :** Utilisation de techniques comme `RandomizedSearchCV` ou `GridSearchCV` pour trouver la combinaison optimale d'hyperparamètres pour le pipeline (nettoyeur, vectoriseur, classifieur) en se basant sur une métrique d'évaluation (`f1_weighted`).
6.  **Évaluation du Modèle :** Le modèle final est évalué sur un ensemble de test indépendant (`X_test`, `y_test`) en utilisant des métriques standard telles que l'exactitude (`accuracy`), le rapport de classification (`classification_report` incluant précision, rappel, F1-score) et la matrice de confusion (`confusion_matrix`).
7.  **Analyse des Coefficients :** Pour les modèles linéaires (Logistic Regression, LinearSVC), une analyse des coefficients associés aux features TF-IDF est réalisée pour identifier les mots ou n-grammes les plus discriminants pour chaque classe (vraies vs fausses nouvelles). Cette analyse permet de comprendre les caractéristiques lexicales associées à chaque catégorie et d'identifier potentiellement des signes de fuite de données si des termes spécifiques et non pertinents ont un poids très élevé.
8.  **Prédiction :** Le pipeline entraîné est utilisé pour prédire la classe de nouveaux articles non vus, fournissant également les probabilités associées à chaque classe.

L'intégralité de ce processus est orchestrée via un pipeline scikit-learn (`sklearn.pipeline.Pipeline`), permettant une gestion cohérente des étapes de prétraitement, vectorisation et classification.

## Fonctionnalités

*   Pipeline de traitement NLP et de classification intégrée.
*   Support de multiples algorithmes de classification texte.
*   Optimisation des hyperparamètres via recherche aléatoire/grille.
*   Évaluation complète du modèle avec les métriques clés.
*   Analyse de l'importance des features pour interpréter le modèle.
*   Prédiction de la véracité avec scores de probabilité.

## Installation

1.  Clonez ce dépôt :
    ```bash
    !git clone https://github.com/Anasviel/Fake_news_detector.git
    ```
2.  Naviguez vers le répertoire du projet :
    ```bash
    cd Fake_news_detector
    ```
3.  Installez les dépendances nécessaires. Il est recommandé d'utiliser un environnement virtuel :
    ```bash
    pip install -r requirements.txt
    ```
    *(Note : Assurez-vous d'avoir un fichier `requirements.txt` listant toutes les bibliothèques utilisées, comme `pandas`, `numpy`, `nltk`, `scikit-learn`, etc. Si vous n'en avez pas, vous devrez l'ajouter en exécutant `pip freeze > requirements.txt` après avoir installé toutes les dépendances nécessaires.)*

## Utilisation

Le projet est principalement implémenté dans un notebook Jupyter (ou Google Colab). Exécutez séquentiellement les cellules du notebook pour :

1.  Charger et inspecter les jeux de données.
2.  Appliquer les fonctions de nettoyage et de prétraitement.
3.  Diviser les données en ensembles d'entraînement et de test.
4.  Définir et configurer le pipeline scikit-learn.
5.  Exécuter l'optimisation des hyperparamètres (si désiré).
6.  Entraîner le modèle final.
7.  Évaluer les performances sur l'ensemble de test.
8.  Analyser l'importance des features (pour les modèles linéaires).
9.  Utiliser le pipeline entraîné pour faire des prédictions sur de nouveaux textes.

## Structure du Projet

*   `notebook.ipynb` (ou `notebook.colab`) : Le notebook principal contenant le code Python pour l'intégralité du pipeline de ML.
*   `WELFake_Dataset.csv` : Le jeu de données principal utilisé pour l'entraînement et les tests.
*   `df_fake_true.csv` : Un jeu de données personnalisé pour les prédictions du modèle sur de nouveaux textes après entraînement.
*   `README.md` : Ce fichier.
*   `LICENSE` : Le fichier de licence du projet.
*   `requirements.txt` (à créer si absent) : Liste des dépendances Python.

## Données

Le projet utilise principalement le jeu de données [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/welfake-dataset) de Kaggle, une collection d'articles de presse labellisés. Un fichier `df_fake_true.csv` personnalisé peut également être utilisé pour tester le modèle sur des exemples spécifiques.

## Modèles Utilisés

Le notebook explore et compare les performances de plusieurs modèles de classification texte, intégrés dans le pipeline après la vectorisation TF-IDF :

*   Multinomial Naive Bayes (avec hyperparamètres optimisés comme `alpha`, `fit_prior`, `class_prior`).
*   Logistic Regression (avec hyperparamètres optimisés comme `C`, `penalty`, `solver`).
*   LinearSVC (avec hyperparamètres optimisés comme `C`).

## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue pour discuter des changements proposés ou soumettre une Pull Request.

## Licence

Ce projet est sous licence [MIT License](LICENSE).

## Auteur

Anasviel
(kevinpages2002@gmail.com)
