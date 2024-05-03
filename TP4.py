import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Données d'entraînement
data = pd.read_csv('train.csv')
data.drop(["Id"], axis=1, inplace=True)

# Divisez les données en variables explicatives (X) et cible (y)
X = data.drop("SalePrice", axis=1)
y = data['SalePrice']

# Gestion des valeurs manquantes et encodage des variables catégorielles
X = pd.get_dummies(X, dummy_na=True)

# Divisez les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialisez et entraînez le modèle d'arbre de décision
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)



# Faites des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculez l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns.tolist(), filled=True, max_depth=3)
plt.show()