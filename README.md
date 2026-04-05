## Projet : Régression linéaire simple – Note des étudiants

## 1. Objectif
L’objectif est de **prédire la note finale d’un étudiant (/20)** en fonction du **nombre d’heures d’étude par jour**.  

Nous utilisons une **régression linéaire simple** :  

## Note= a ⋅ Heures_etude + b

- **a (pente)** : indique combien la note augmente pour chaque heure d’étude supplémentaire  
- **b (ordonnée à l’origine)** : note estimée si l’étudiant n’étudie pas  

Ainsi, les **deux paramètres du modèle** sont `a` et `b`.

---

## 2. Dataset
Le dataset utilisé se trouve dans :  
`data/etudiant.csv`

### Variables :

| Nom de la variable | Description |
|------------------|-------------|
| Heures_etude     | Nombre d’heures d’étude par jour |
| Note             | Note finale de l’étudiant (/20) |

💡 Ces données sont construites pour **comprendre le fonctionnement d’une régression linéaire simple**.

---

- On estime **a** et **b** à partir des données  
- Ce modèle permet de **prévoir la note à partir du nombre d’heures d’étude**  
- C’est une approche simple mais pédagogique pour comprendre la régression linéaire

