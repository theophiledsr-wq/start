import streamlit as st
import pandas as pd
import numpy as np

st.title("Mon Analyseur de Données")

# Création du formulaire
with st.form("mon_formulaire"):
    nom = st.text_input("Entrez votre nom")
    nombre_points = st.slider("Choisissez le nombre de points", 10, 100, 50)
    couleur = st.color_picker("Choisissez une couleur pour le graphique", "#00f900")
    
    # Bouton de soumission
    submitted = st.form_submit_button("Générer le graphique")

if submitted:
    st.write(f"Bonjour {nom} ! Voici vos données :")
    
    # Génération de données aléatoires
    chart_data = pd.DataFrame(
        np.random.randn(nombre_points, 2),
        columns=['A', 'B']
    )
    
    # Affichage du graphique
    st.line_chart(chart_data, color=couleur)
