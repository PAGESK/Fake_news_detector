# 📰 Fake News Detector (NLP + Machine Learning + Explainability)

## **Description**
Ce projet implémente un modèle de Machine Learning pour la **détection automatique de fake news** basé sur :
- **TF-IDF** pour la vectorisation des textes.
- **Régression logistique** pour la classification.
- **LIME** pour l’explicabilité locale.
- **SHAP** pour l’explicabilité globale (optionnel).

Le projet est pensé pour être **reproductible, clair et modulaire** :
- Un **notebook explicatif** pour comprendre l’ensemble du pipeline.
- Des **scripts indépendants** pour l’entraînement et les prédictions.
- Une **gestion des dépendances** avec des versions figées pour éviter les incompatibilités.



## **Installation**

### **Option 1 – Environnement complet (Colab)**
```bash
pip install -r requirements.txt
```

### **Option 2 – Environnement minimal**
```
pip install -r requirements_project.txt
```
### **Option 3 – Sur Google Colab**

- Ouvrir le notebook main_fake_news_detector.ipynb.

- Vérifier que requirements.txt est bien installé :
```
!pip install -r requirements.txt
```
## **Utilisation
1. Exécuter le notebook complet

Le notebook main_fake_news_detector.ipynb contient :

Nettoyage des données

Entraînement du modèle

Évaluation

Visualisation des métriques

Explicabilité LIME





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
