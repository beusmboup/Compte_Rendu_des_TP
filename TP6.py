import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Charger les données
data = pd.read_csv('C:/Users/KHALIFA/PycharmProjects/TestDeDonnées/Social_Network_Ads.csv')

# Sélectionner les features
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Mise à l'échelle des features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialiser le modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Entraîner le modèle
knn.fit(X_train, y_train)


# Fonction pour tracer la frontière de décision
def plot_decision_boundary(X, y, model, scaler, h=0.02):
    # Définir les limites des axes x et y
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Créer la grille de points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Prétraiter les données avec le scaler
    X_scaled = scaler.transform(np.c_[xx.ravel(), yy.ravel()])

    # Prédire les étiquettes pour chaque point dans la grille
    Z = model.predict(X_scaled)

    # Remettre la forme de Z à la grille
    Z = Z.reshape(xx.shape)

    # Tracer les contours de décision
    plt.contourf(xx, yy, Z, alpha=0.4)

    # Tracer les points de données
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.title('Decision Boundary')
    plt.show()


# Tracer la frontière de décision
plot_decision_boundary(X_train, y_train, knn, scaler)
