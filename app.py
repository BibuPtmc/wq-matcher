import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

st.title("Démo Explicabilité - LIME & Activation Map (ResNet18)")

st.write("""
Téléchargez une image (ou donnez une URL) pour voir :
- L'explication LIME (zones importantes pour l'embedding)
- La carte d'activation principale de ResNet18
""")

# Entrée image
img_file = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"])
img_url = st.text_input("Ou collez une URL d'image :")

if img_file:
    image = Image.open(img_file).convert("RGB")
    # On l'upload sur un endpoint temporaire (ou on l'encode en base64 pour l'envoyer)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    # On peut utiliser un service d'upload si besoin, ici on va juste l'afficher
    st.image(image, caption="Image uploadée", use_column_width=True)
    st.warning("Pour la démo API, utilisez l'URL d'une image accessible publiquement (Cloudinary, etc.)")
elif img_url:
    try:
        response = requests.get(img_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(image, caption="Image depuis l'URL", use_column_width=True)
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image : {e}")
        image = None
else:
    image = None

if image and img_url:
    with st.spinner("Appel de l'API pour LIME et Activation Map..."):
        # Appel LIME
        lime_resp = requests.post("http://localhost:8000/lime", json={"url": img_url})
        if lime_resp.status_code == 200:
            lime_b64 = lime_resp.json()["lime_explanation"]
            lime_img = Image.open(BytesIO(base64.b64decode(lime_b64)))
        else:
            lime_img = None
            st.error("Erreur LIME : " + lime_resp.text)
        # Appel Activation Map
        act_resp = requests.post("http://localhost:8000/activation-map", json={"url": img_url})
        if act_resp.status_code == 200:
            act_b64 = act_resp.json()["activation_map"]
            act_img = Image.open(BytesIO(base64.b64decode(act_b64)))
        else:
            act_img = None
            st.error("Erreur Activation Map : " + act_resp.text)
    # Affichage côte à côte
    st.subheader("Résultats :")
    cols = st.columns(3)
    cols[0].image(image, caption="Image originale", use_column_width=True)
    if lime_img:
        cols[1].image(lime_img, caption="LIME", use_column_width=True)
    if act_img:
        cols[2].image(act_img, caption="Activation Map", use_column_width=True)
    st.success("Démo terminée !")
elif image and not img_url:
    st.info("Pour la démo API, veuillez fournir une URL d'image accessible publiquement.") 