from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
y = data.target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialiser et entraîner le modèle de l'arbre de décision
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Afficher l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

