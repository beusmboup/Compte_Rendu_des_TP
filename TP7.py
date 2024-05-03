import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
# Charger les données depuis un fichier CSV
data = pd.read_csv('C:/Users/KHALIFA/PycharmProjects/TestDeDonnées/student_admission_dataset.csv')

# Diviser les données en caractéristiques (X) et étiquettes de classe (y)
X = data[['GPA', 'SAT_Score']].values
y = data['Admission_Status'].values

# Convertir les étiquettes de classe en valeurs numériques
le = LabelEncoder()
y = le.fit_transform(y)

# Entraîner un modèle SVM sur l'ensemble de données complet
svm_model = SVC(kernel='linear')
svm_model.fit(X, y)

# Générer des prédictions sur l'ensemble de données complet
predictions = svm_model.predict(X)

# Créer un DataFrame pour stocker les prédictions et les scores
predictions_df = pd.DataFrame(X, columns=['GPA', 'SAT_Score'])
predictions_df['Admission_Status'] = le.inverse_transform(predictions)

# Trier les prédictions par ordre croissant des scores SAT
predictions_df = predictions_df.sort_values(by='SAT_Score')

# Diviser les prédictions en étudiants admis et refusés
admitted_students = predictions_df[predictions_df['Admission_Status'] == 'Accepted']
rejected_students = predictions_df[predictions_df['Admission_Status'] == 'Rejected']

# Trouver les coefficients du modèle SVM
coef = svm_model.coef_[0]
intercept = svm_model.intercept_

# Calculer les coordonnées de la ligne diagonale entre les admis et les refusés
x_values = np.linspace(0, 4, 100)
y_values = (-coef[0] / coef[1]) * x_values - (intercept[0] / coef[1])

# Afficher les prédictions avec la diagonale
plt.figure(figsize=(10, 6))
plt.scatter(rejected_students['GPA'], rejected_students['SAT_Score'], c='red', marker='x', label='Rejected')
plt.scatter(admitted_students['GPA'], admitted_students['SAT_Score'], c='blue', marker='o', edgecolors='k', label='Admitted')

# Remplir les zones de couleur
plt.fill_between(x_values, y_values, np.min(predictions_df['SAT_Score']) - 0.1, color='red', alpha=0.1)
plt.fill_between(x_values, y_values, np.max(predictions_df['SAT_Score']) + 0.1, color='blue', alpha=0.1)

# Tracer la diagonale
plt.plot(x_values, y_values, color='black', linestyle='--', linewidth=2)
plt.xlabel('GPA')
plt.ylabel('SAT Score')
plt.title('Predictions - Student Admission Dataset')
plt.legend()
plt.show()