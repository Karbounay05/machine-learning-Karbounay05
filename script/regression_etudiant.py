# ===============================
# Régression linéaire simple – Note des étudiants
# ===============================

# ------------------------------
# 1️⃣ Import des bibliothèques
# ------------------------------
import pandas as pd                    # Pour manipuler les données
import matplotlib.pyplot as plt        # Pour tracer des graphiques
from sklearn.linear_model import LinearRegression  # Pour créer le modèle de régression
from sklearn.metrics import mean_squared_error, r2_score  # Pour évaluer le modèle

# ------------------------------
# 2️⃣ Charger le dataset
# ------------------------------
# Assurez-vous que le fichier CSV est bien dans "dataset/etudiant.csv"
df = pd.read_csv("dataset/etudiant.csv")

# Vérifier les 5 premières lignes
print("Aperçu du dataset :")
print(df.head())

# ------------------------------
# 3️⃣ Visualiser la relation entre Heures d'étude et Note
# ------------------------------
plt.scatter(df['Heures_etude'], df['Note'], color='blue')
plt.xlabel("Heures d'étude par jour")
plt.ylabel("Note finale (/20)")
plt.title("Relation entre Heures d'étude et Note")
plt.show()

# ------------------------------
# 4️⃣ Préparer les données pour le modèle
# ------------------------------
X = df[['Heures_etude']]  # variable indépendante (2D)
y = df['Note']            # variable dépendante

# ------------------------------
# 5️⃣ Créer et entraîner le modèle
# ------------------------------
model = LinearRegression()
model.fit(X, y)

# Récupérer les paramètres du modèle
a = model.coef_[0]      # pente
b = model.intercept_    # ordonnée à l'origine
print(f"Modèle entraîné : Note = {a:.2f} * Heures_etude + {b:.2f}")

# ------------------------------
# 6️⃣ Prédictions et évaluation
# ------------------------------
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Erreur quadratique moyenne (MSE) : {mse:.2f}")
print(f"Coefficient de détermination (R²) : {r2:.2f}")

# ------------------------------
# 7️⃣ Visualiser la droite de régression
# ------------------------------
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, y_pred, color='red', label='Droite de régression')
plt.xlabel("Heures d'étude par jour")
plt.ylabel("Note finale (/20)")
plt.title(f"Régression linéaire simple : Note = {a:.2f}*Heures_etude + {b:.2f}")
plt.legend()
plt.show()