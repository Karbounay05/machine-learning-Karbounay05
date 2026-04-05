[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/jyj5d-5u)
code python de Régession Linèaire, Multiple et polynomiale (From scratch et avec bibliothèque)...


-------------------------------------------------------------------------------------------

# Dataset : Performance des étudiants

## Objectif

Dans ce travail, on cherche à prédire la note finale d’un étudiant sur 20 en fonction de plusieurs facteurs liés à ses habitudes (étude, sommeil, stress, etc.).

---

## Description des variables

* Heures d’étude : nombre d’heures d’étude par jour
* Sommeil : nombre d’heures de sommeil par nuit
* Stress : niveau de stress (entre 1 et 10)
* Participation : participation en classe (en %)
* Activité sportive : nombre d’heures de sport par semaine
* Temps écran : temps passé sur les écrans (heures par jour)
* Note finale : note obtenue par l’étudiant (/20)

---

## Données

| Heures étude | Sommeil | Stress | Participation | Sport | Temps écran | Note |
| ------------ | ------- | ------ | ------------- | ----- | ----------- | ---- |
| 2            | 5       | 8      | 20            | 0     | 6           | 10   |
| 4            | 6       | 6      | 40            | 2     | 5           | 13   |
| 6            | 7       | 5      | 60            | 3     | 4           | 15   |
| 8            | 7       | 4      | 80            | 4     | 3           | 17   |
| 10           | 8       | 3      | 90            | 5     | 2           | 18.5 |
| 3            | 4       | 9      | 30            | 0     | 7           | 11   |
| 7            | 6       | 5      | 70            | 3     | 4           | 16   |

---

## Modèle

On utilise une régression linéaire multiple pour modéliser la relation entre les variables :

Note = a₁·Heures + a₂·Sommeil + a₃·Stress + a₄·Participation + a₅·Sport + a₆·Ecran + b

---

## Remarque

Ces données sont des exemples construits pour comprendre le fonctionnement des modèles de régression et tester différentes approches (linéaire, multiple et polynomiale).

---
