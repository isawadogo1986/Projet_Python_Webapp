#CREATION D'UN MODELE PREDICTIVE DU NOMBRE DES TRAVAILLEURS DE L'ENTREPRISE DURANT UN MOIS REGULIER
#Installation des bibliothèques
## Importation des Bibliothéques requises
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from sklearn import datasets
from sklearn.linear_model import LinearRegression

st.write('''
# Bienvenue Hello
'''
)

#Importation des données d'enquête sur les entreprises informelles
# Importation des données Chemin vers votre fichier .dta
df = pd.read_stata("C:/Users/User/Documents/Zoom/Candidature 2025/Formation Sorbonne Data Analytics/Introduction to Python/Projet python/assets/Informal-Sector-Enterprise-Surveys-Indicators-Database_February_3_2025.dta")
# Description des données
df.describe()
df.info()
#Simuler l'entraînement du modèle (à remplacer par votre modèle réel)
def train_worker_prediction_model():

    X= df[['dem6', 'dem7', 'op9']]
    y = df['wf1'] 
    model = LinearRegression()
    model.fit(X, y)
    return model

# Sauvegarder le modèle simulé
model = train_worker_prediction_model()
pickle.dump(model, open('modele_prediction_travailleurs.pkl', 'wb'))
# --- 2. Fonction de prédiction ---
def predict_workers(model, demo6, demo7, op9):
    input_data = np.array([[demo6, demo7, op9]])
    prediction = model.predict(input_data)
    return max(0, int(round(prediction[0]))) # S'assurer que la prédiction n'est pas négative
# --- 3. Interface Streamlit ---
st.title("Prédiction du Nombre de Travailleurs par Mois")

st.subheader("Veuillez entrer les caractéristiques de l'entreprise :")

demo6_input = st.number_input("Âge moyen de l'entreprise (en années)", min_value=0.0, step=1.0)
demo7_input = st.number_input("Âge moyen du propriétaire (en années)", min_value=0.0, step=1.0)
op9_input = st.number_input("Nombre d'heures d'ouverture par semaine", min_value=0.0, step=1.0)

# Charger le modèle pré-entraîné
loaded_model = pickle.load(open('modele_prediction_travailleurs.pkl', 'rb'))

if st.button("Prédire le nombre de travailleurs"):
    prediction = predict_workers(loaded_model, demo6_input, demo7_input, op9_input)
    st.subheader(f"Le nombre de travailleurs prédit pour le mois est : {prediction}")

st.sidebar.header("À propos de l'application")
st.sidebar.info("Cette application prédit le nombre de travailleurs par mois en fonction de l'âge moyen de l'entreprise, de l'âge moyen du propriétaire et du nombre d'heures d'ouverture par semaine.")
#help(pd.read_csv)
#df.info()
print(df.head())
df.info()
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

# --- Simuler l'entraînement d'un modèle (à remplacer par votre modèle réel) ---
def train_house_price_model():
    superficie = np.array([50, 75, 100, 125, 150]).reshape(-1, 1)
    prix = np.array([150000, 225000, 300000, 375000, 450000])
    model = LinearRegression()
    model.fit(superficie, prix)
    return model

# Sauvegarder le modèle simulé
model = train_house_price_model()
pickle.dump(model, open('modele_prix_maison.pkl', 'wb'))

# --- Fonction de prédiction ---
def predict_house_price(model, superficie):
    prediction = model.predict(np.array([[superficie]]))
    return prediction[0]

# --- Interface Streamlit ---
st.title("Prédicteur de Prix de Maison")

st.subheader("Entrez la superficie de la maison (en m²) :")
superficie_input = st.number_input("Superficie", min_value=1.0)

# Charger le modèle pré-entraîné
loaded_model = pickle.load(open('modele_prix_maison.pkl', 'rb'))

if st.button("Prédire le Prix"):
    prix_predit = predict_house_price(loaded_model, superficie_input)
    st.subheader(f"Le prix de la maison prédit est : {prix_predit:.2f} FCFA") # Adaptation de l'unité monétaire

st.sidebar.header("À propos de l'application")
st.sidebar.info("Cette application prédit le prix d'une maison en fonction de sa superficie, en utilisant un modèle de régression linéaire simple.")