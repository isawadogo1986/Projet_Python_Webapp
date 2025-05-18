# Projet_Python_Webapp
L'objectif de la webapp:
L’objectif de cette web app est de prédire l’indice de capacité de gestion (variant de 0 à 100) des entreprises informelles des pays en développement. Ils’agit d’un indice composité qui englobe plusieurs aspects liés aux pratiques de gestion de ces entreprises. Cet indice varie de 0 à 100. Les entreprises ayant un indice tendant vers 100 ont une meilleure capacité de gestion contrairement à celles ayant un indice tendant vers 0.

Le choix du dataset:
Les données utilisées proviennent de la Banque Mondiale; plus précisément de la base de données «Informal-Sector-Enterprise-Surveys-Indicators» qui sont des données open source.

Le choix du modèle:
Nous avons utilisé un modèle de régression linéaire pour expliquer l’indice de capacité de gestion des entreprises du secteur informel dans les pays en développement. Le modèle contient dix variables explicatives à savoir: le pays d'appartenance de l'entreprise, le niveau d'étude secondaire du propriétaire principal (0=Non, 100=Oui), l'âge de l'entreprise, l'âge du propriétaire ou âge moyen des propriétaires, le nombre moyen d'employés, le sexe du propriétaire ou des propriétaires principales (0=Homme, 100=Femme), les entreprises enregistrant un profit le mois passé (0=Non, 100=Oui), le nombre moyen d'heures d'ouverture dans la semaine, les entreprises utilisant le mobile money (0=Non, 100=Oui), les entreprises utilisant un ordinateur/tablette (0=Non, 100=Oui) ."

Le fonctionnement global de l'application:
Sur l’interface de l’application, nous avons un avant-propos qui explique l’objectif de l’application et son fonctionnement. Nous avons également les entrées qui permettent de renseigner les caractéristiques d’une entreprise quelconque afin de prédire son indice de capacité de gestion. Enfin un bouton permettant de lancer la prédiction après avoir renseigné les différents caractéristques de l’entreprise choisie.
