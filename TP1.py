import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Chargement des données
mpg_data = pd.read_csv("auto-mpg.csv")

# Remplacement des valeurs non numériques par la moyenne
mpg_data["horsepower"] = pd.to_numeric(mpg_data["horsepower"], errors="coerce")
mpg_data["horsepower"].fillna(mpg_data["horsepower"].mean(), inplace=True)

# Sélection de la variable cible (consommation de carburant)
y = mpg_data["mpg"]

# Sélection de la variable prédictive (par exemple, la puissance)
X = mpg_data[["horsepower"]]

# Entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Préparation des données pour la visualisation
x_values = X.values.reshape(-1, 1)
y_pred = model.predict(x_values)

# Affichage de la courbe de régression linéaire
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(x_values, y_pred, color='red', linewidth=2, label='Régression linéaire')
plt.xlabel('Puissance')
plt.ylabel('Consommation de carburant (MPG)')
plt.title('Régression linéaire sur le dataset MPG')
plt.legend()
plt.show()
