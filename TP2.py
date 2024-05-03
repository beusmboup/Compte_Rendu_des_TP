from sklearn.datasets import load_breast_cancer
import pandas as pd
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

data = load_breast_cancer()
X = data.data[:, :2]  # Utiliser uniquement les deux premières fonctionnalités pour une visualisation bidimensionnelle
y = data.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialiser et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Tracer la courbe de décision
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=model, legend=2)
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[7])
plt.title('Courbe de décision pour le cancer du sein')
plt.show()