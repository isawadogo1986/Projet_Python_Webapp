# MODELE PREDICTIVE DE L'INDICE DE LA CAPACITÉ DE GESTION DES ENTREPRISES DANS LES PAYS EN DÉVELOPPEMENT
#Installation des librairies
## Importation des Bibliothéques requises
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#Importation des données d'enquête sur les entreprises informelles
# Importation des données Chemin vers votre fichier .dta
df = pd.read_stata("C:/Users/User/Documents/Zoom/Candidature 2025/Formation Sorbonne Data Analytics/Introduction to Python/Projet python/assets/Informal-Sector-Enterprise-Surveys-Indicators-Database_February_3_2025.dta")
#df.describe()
#df.info()
# Encodage de la variable 'Pays'
label_encoder=LabelEncoder()
df['Country_Encoded'] = label_encoder.fit_transform(df['country'])
#df.describe()
#import pickle as pkl
#Simulation de l'entraînement du modèle
def train_management_index_prediction_model():

    X= df[['Country_Encoded','dem2','dem6','dem7','wf1','ge1','op1','op9','tech2','tech3']]
    y = df['mg1'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


#Sauvegarde du modèle simulé
# Sauvegarder le modèle simulé
model = train_management_index_prediction_model()
pickle.dump(model, open('modèle de prédiction de la capacité de gestion des entreprises.pkl', 'wb'))
# --- 2. Fonction de prédiction ---
def predict_management_index(Country_Encoded,dem2,dem6,dem7,wf1,ge1,op1,op9,tech2,tech3):
    input_data = np.array([[Country_Encoded,dem2,dem6,dem7,wf1,ge1,op1,op9,tech2,tech3]])
    prediction = model.predict(input_data)
    return max(0, int(round(prediction[0]))) # S'assurer que la prédiction n'est pas négative
#Interface Streamlit
# --- 3. Interface Streamlit ---
st.title("Prédiction de l'indice de la capacité de gestion de l'entreprise")

st.subheader("Veuillez entrer les caractéristiques de l'entreprise :")
# Liste déroulante pour le pays
country_options = df['country'].unique()
country_choisi = st.selectbox("Pays d'appartenance de l'entreprise", country_options)
Country_Encoded = label_encoder.transform([country_choisi])[0]
# Input pour le niveau d'étude secondaire
dem2 = st.slider("Niveau d'étude secondaire du propriétaire principal (0=Non, 100=Oui)", 0, 100, 100, step=100)
# Input pour l'âge de l'entreprise
dem6 = st.number_input("Âge de l'entreprise", min_value=0, step=1, value=0)
# Input pour l'âge moyen des propriétaires
dem7 = st.number_input("Âge du propriétaire ou âge moyen des propriétaires", min_value=17, step=1, value=17)
# Input pour le nombre moyen d'employés
wf1 = st.number_input("Nombre moyen d'employés", min_value=0, step=1, value=0)
# Input pour le sexe féminin des propriétaires
ge1 = st.slider("Le sexe du proprétaire ou des propriétaires principaux est féminin (0=Non, 100=Oui)", 0, 100, 0, step=100)
# Input pour l'enregistrement de profit
op1 = st.slider("Les entreprises enregistrant un profit le mois passé (0=Non, 100=Oui)", 0, 100, 100, step=100)
# Input pour le nombre moyen d'heures d'ouverture
op9= st.number_input("Nombre moyen d'heures d'ouverture dans la semaine", min_value=0, step=1, value=40)
# Input pour l'utilisation du mobile money
tech2= st.slider("Les entreprises utilisant le mobile money (0=Non, 100=Oui)", 0, 100, 100, step=100)
# Input pour l'utilisation d'un ordinateur/tablette
tech3= st.slider("Les entreprises utilisant un ordinateur/tablette (0=Non, 100=Oui)", 0, 100, 0, step=100)
# Bouton de prédiction
if st.button("Prédire l'Indice de la capacité de Gestion"):
    input_data = pd.DataFrame({
        'Country_Encoded': [Country_Encoded],
        'dem2': [dem2],
        'dem6': [dem6],
        'dem7': [dem7],
        'wf1': [wf1],
        'ge1': [ge1],
        'op1': [op1],
        'op9': [op9],
        'tech2': [tech2],
        'tech3': [tech3]
    })
    prediction = model.predict(input_data)
    st.subheader(f"L'indice de capacité de gestion prédit est : {prediction[0]:.2f}")

st.caption("Ceci est un modèle de prédiction basé sur les données fournies.")
st.sidebar.header("À propos de l'application")
st.sidebar.info("Cette application prédit L'indice de capacité de gestion(variant de 0 à 100) des entreprises informelles de pays en développement, selon le Pays d'appartenance de l'entreprise, le niveau d'étude secondaire du propriétaire principal (0=Non, 100=Oui),l'âge de l'entreprise, l'âge du propriétaire ou âge moyen des propriétaires, le nombre moyen d'employés, Le sexe du proprétaire ou des propriétaires principales (0=Homme, 100=Femme),les entreprises enregistrant un profit le mois passé (0=Non, 100=Oui),le nombre moyen d'heures d'ouverture dans la semaine, les entreprises utilisant le mobile money (0=Non, 100=Oui), Les entreprises utilisant un ordinateur/tablette (0=Non, 100=Oui) .")

#dem2_input = st.number_input("Le propriétaire principal a terminé les études sécondaires", min_value=0.0, step=1.0)
#dem6_input = st.number_input("Âge moyen de l'entreprise (en années)", min_value=0.0, step=1.0)
#dem7_input = st.number_input("Âge moyen du propriétaire (en années)", min_value=0.0, step=1.0)
#wf1_input = st.number_input("Nombre moyen d'employés de l'entreprise (en années)", min_value=0.0, step=1.0)
#ge1_input = st.number_input("Les propriétaires principales sont des femmes", min_value=0.0, step=1.0)
#op1_input = st.number_input("Les entreprises enregistrant un profit le mois passé", min_value=0.0, step=1.0)
#op9_input = st.number_input("Nombre moyen d'heures d'ouverture de la semaine ", min_value=0.0, step=1.0)
#tech2_input = st.number_input("Les entreprises utilisant le mobile money", min_value=0.0, step=1.0)
#tech3_input = st.number_input("Les entreprises utilisant un ordinataire/tablette", min_value=0.0, step=1.0)
# Charger le modèle pré-entraîné
#loaded_model = pickle.load(open('modèle de prédiction de la capacité de gestion.pkl','rb'))

#if st.button("Prédire l'index de capacité ge gestion"):
    #prediction = predict_workers(loaded_model, demo6_input, demo7_input, op9_input)
    #st.subheader(f"L'indice de capacité de gestion est : {prediction}")

#st.sidebar.header("À propos de l'application")
#st.sidebar.info("Cette application prédit le nombre de travailleurs par mois en fonction de l'âge moyen de l'entreprise, de l'âge moyen du propriétaire et du nombre d'heures d'ouverture par semaine.")
#help(pd.read_csv)
#df.info()
#print(df.head())
#df.info()
#import streamlit as st
#import pandas as pd
#from sklearn.linear_model import LinearRegression
#import numpy as np
#import pickle

