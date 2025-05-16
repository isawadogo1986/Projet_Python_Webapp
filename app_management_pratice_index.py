#CREATION D'UN MODELE PREDICTIVE DU NOMBRE DES TRAVAILLEURS DE L'ENTREPRISE DURANT UN MOIS REGULIER
#Installation des bibliothèques
## Importation des Bibliothéques requises
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#Importation des données d'enquête sur les entreprises informelles
# Importation des données Chemin vers votre fichier .dta
df = pd.read_stata("C:/Users/User/Documents/Zoom/Candidature 2025/Formation Sorbonne Data Analytics/Introduction to Python/Projet python/assets/Informal-Sector-Enterprise-Surveys-Indicators-Database_February_3_2025.dta")
# Description des données
#df.describe()
#df.info()
#df.describe()
#Simuler l'entraînement du modèle (à remplacer par votre modèle réel)
def train_management_index_prediction_model():

    X= df[['dem2','dem6','dem7','wf1','ge1','op1','op9','tech2','tech3']]
    y = df['mg1'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Sauvegarder le modèle simulé
model = train_management_index_prediction_model()
pickle.dump(model, open('modèle de prédiction de la capacité de gestion.pkl', 'wb'))
# --- 2. Fonction de prédiction ---
def predict_workers(dem2,dem6,dem7,wf1,ge1,op1,op9,tech2,tech3):
    input_data = np.array([[dem2,dem6,dem7,wf1,ge1,op1,op9,tech2,tech3]])
    prediction = model.predict(input_data)
    return max(0, int(round(prediction[0]))) # S'assurer que la prédiction n'est pas négative
# --- 3. Interface Streamlit ---
st.title("Prédiction de la capacité de la pratique de gestion de l'entreprise")

st.subheader("Veuillez entrer les caractéristiques de l'entreprise :")

dem2_input = st.number_input("Le propriétaire principal a terminé les études sécondaires", min_value=0.0, step=1.0)
dem6_input = st.number_input("Âge moyen de l'entreprise (en années)", min_value=0.0, step=1.0)
dem7_input = st.number_input("Âge moyen du propriétaire (en années)", min_value=0.0, step=1.0)
wf1_input = st.number_input("Nombre moyen d'employés de l'entreprise (en années)", min_value=0.0, step=1.0)
ge1_input = st.number_input("Les propriétaires principales sont des femmes", min_value=0.0, step=1.0)
op1_input = st.number_input("Les entreprises enregistrant un profit le mois passé", min_value=0.0, step=1.0)
op9_input = st.number_input("Nombre moyen d'heures d'ouverture de la semaine ", min_value=0.0, step=1.0)
tech2_input = st.number_input("Les entreprises utilisant le mobile money", min_value=0.0, step=1.0)
tech3_input = st.number_input("Les entreprises utilisant un ordinataire/tablette", min_value=0.0, step=1.0)
# Charger le modèle pré-entraîné
loaded_model = pickle.load(open('modèle de prédiction de la capacité de gestion.pkl','rb'))

if st.button("Prédire l'index de capacité ge gestion"):
    prediction = predict_workers(loaded_model, demo6_input, demo7_input, op9_input)
    st.subheader(f"L'indice de capacité de gestion est : {prediction}")

st.sidebar.header("À propos de l'application")
st.sidebar.info("Cette application prédit l'indice de la capacité de gestion des entreprises selon dem2 dem6 dem7 wf1 ge1 tech2 tech3 op1 op9.")
