# Importation de pandas pour lire les fichiers CSV
import pandas as pd
# Importation de math pour utiliser les fonctions mathematiques
import math

# Creation de la classe Naive Bayes
class NaiveBayesStudentModel:
    # Constructeur de la classe
    def __init__(self):
        # Initialisation des dictionnaires
        self.classes = {}
        self.statistiques = {}
        self.probabilites_classes = {}

    # Fonction pour charger les donnees
    def load_data(self, file_path):
        # Lecture du fichier CSV
        data = pd.read_csv(file_path)
        # Retour des donnees
        return data

    # Fonction pour preparer les donnees
    def prepare_data(self, data):
        # Selection des colonnes d'entree
        X = data[["heures_etudes", "participation"]]
        # Selection de la colonne resultat
        y = data["resultat"]
        # Retour des donnees
        return X, y

    # Fonction pour entrainer le modele
    def train(self, X, y):
        # Recuperation des classes uniques
        classes = y.unique()
        # Boucle sur chaque classe
        for classe in classes:
            # Filtrage des lignes correspondant a la classe
            lignes = X[y == classe]
            # Calcul de la moyenne des heures d'etudes
            moyenne_heures = lignes["heures_etudes"].mean()
            # Calcul de l'ecart type des heures d'etudes
            ecart_heures = lignes["heures_etudes"].std()
            # Calcul de la moyenne de participation
            moyenne_participation = lignes["participation"].mean()
            # Calcul de l'ecart type de participation
            ecart_participation = lignes["participation"].std()
            # Sauvegarde des statistiques
            self.statistiques[classe] = {
                "moyenne_heures": moyenne_heures,
                "ecart_heures": ecart_heures,
                "moyenne_participation": moyenne_participation,
                "ecart_participation": ecart_participation
            }
            # Calcul de la probabilite de la classe
            self.probabilites_classes[classe] = len(lignes) / len(X)
        # Message de confirmation
        print("Modele entraine avec succes")

    # Fonction de calcul de probabilite gaussienne
    def gaussian_probability(self, x, moyenne, ecart_type):
        # Verification que l'ecart type n'est pas nul
        if ecart_type == 0:
            ecart_type = 0.0001
        # Calcul de l'exposant
        exposant = math.exp(-((x - moyenne) ** 2) / (2 * (ecart_type ** 2)))
        # Calcul de la formule gaussienne
        resultat = (1 / (math.sqrt(2 * math.pi) * ecart_type)) * exposant
        # Retour du resultat
        return resultat

    # Fonction de prediction
    def predict_student(self, heures_etudes, participation):
        # Creation d'un dictionnaire pour stocker les probabilites
        probabilites = {}
        # Boucle sur chaque classe
        for classe in self.statistiques:
            # Recuperation des statistiques
            moyenne_heures = self.statistiques[classe]["moyenne_heures"]
            ecart_heures = self.statistiques[classe]["ecart_heures"]
            moyenne_participation = self.statistiques[classe]["moyenne_participation"]
            ecart_participation = self.statistiques[classe]["ecart_participation"]
            # Calcul de la probabilite des heures
            prob_heures = self.gaussian_probability(heures_etudes, moyenne_heures, ecart_heures)
            # Calcul de la probabilite de participation
            prob_participation = self.gaussian_probability(participation, moyenne_participation, ecart_participation)
            # Calcul de la probabilite finale
            probabilite_finale = self.probabilites_classes[classe] * prob_heures * prob_participation
            # Sauvegarde de la probabilite
            probabilites[classe] = probabilite_finale
        # Recherche de la classe ayant la plus grande probabilite
        prediction = max(probabilites, key=probabilites.get)
        # Retour de la prediction
        return prediction, probabilites