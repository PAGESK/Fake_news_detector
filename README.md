# 📰 Fake News Detector (NLP + Machine Learning + Explainability)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PAGESK/FakeNewsDetector/blob/main/main_fake_news_detector.ipynb)


## **Description**
Ce projet implémente un modèle de Machine Learning pour la **détection automatique de fake news** basé sur :
- **TF-IDF** pour la vectorisation des textes.
- **Régression logistique** pour la classification.
- **LIME** pour l’explicabilité locale.
-

Le projet est pensé pour être **reproductible, clair et modulaire** :
- Un **notebook explicatif** pour comprendre l’ensemble du pipeline.
- Des **scripts indépendants** pour l’entraînement et les prédictions.
- Une **gestion des dépendances** avec des versions figées pour éviter les incompatibilités.



## **Installation**

1. Ouvrez le fichier *main_fake_news_detector.ipynb* sur Google Colab (ou cliquez directement sur le badge "Open in Colab" ci-dessus).
2. Suivez les instructions contenues dans le notebook et exécutez les cellules. Le repository github se copie automatiquement sur colab et les dépendances nécessaires sont installées au début du notebook.


Le notebook *main_fake_news_detector.ipynb* contient :

- Nettoyage des données

- Entraînement du modèle

- Évaluation

- Visualisation des métriques

- Explicabilité LIME

- Focus sur l'éthique et l'EU AI Act

## **Aperçut des résultats :**


**Performance initiale sur le dataset WELFake :**

<img width="703" height="246" alt="{5733A4A0-9DFF-4D38-870F-01FA8B1F3778}" src="https://github.com/user-attachments/assets/497d6ca9-9025-42e3-8c76-61786a403b8c" />



**Améliorations dans la seconde version :**

- Correction du data leakage
- Randomized Grid Search pour réduire l’overfitting et améliorer la généralisabilité
  
<img width="725" height="636" alt="image" src="https://github.com/user-attachments/assets/3dc8d90a-1cff-4d23-8b83-5e6966405bd8" />

<img width="544" height="253" alt="{D7864360-9559-4B34-B62F-3233418E2692}" src="https://github.com/user-attachments/assets/91072ff9-9f4b-4e26-9d28-386cdb2b7664" />






## Données

Le projet utilise le jeu de données [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/welfake-dataset) de Kaggle, une collection d'articles de presse labellisés. Un fichier `df_fake_true.csv` créé manuellement peut également être utilisé pour tester le modèle sur des exemples spécifiques.

## Modèles Utilisés

Le notebook explore et compare les performances de plusieurs modèles de classification texte, intégrés dans le pipeline après la vectorisation TF-IDF :

*   Multinomial Naive Bayes 
*   Logistic Regression 
*   LinearSVC


## Versions


## Contribution

Les contributions sont les bienvenues ! Veuillez ouvrir une issue pour discuter des changements proposés ou soumettre une Pull Request.

## Licence

Ce projet est sous licence [MIT License](LICENSE).

## Auteur

Kevin
(kevinpages2002@gmail.com)
