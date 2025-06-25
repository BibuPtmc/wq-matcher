from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import numpy as np
from typing import List
import logging
import base64
import lime
from lime import lime_image
from skimage.transform import resize
from skimage import feature

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Embedding Service")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle ResNet18 pré-entraîné
model = resnet18(weights=ResNet18_Weights.DEFAULT)
# Suppression de la dernière couche (classification)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class ImageUrl(BaseModel):
    url: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class LimeExplanationResponse(BaseModel):
    lime_explanation: str  # image LIME encodée en base64

class ActivationMapResponse(BaseModel):
    activation_map: str  # image heatmap encodée en base64

def get_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de l'image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Impossible de télécharger l'image: {str(e)}")

def get_embedding(image: Image.Image) -> List[float]:
    try:
        # Prétraitement de l'image
        image_tensor = transform(image).unsqueeze(0)
        
        # Génération de l'embedding
        with torch.no_grad():
            embedding = model(image_tensor)
            embedding = embedding.squeeze().numpy()
            
        # Normalisation de l'embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Erreur lors de la génération de l'embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'embedding: {str(e)}")

def compute_lime_explanation(image: Image.Image, model, transform) -> str:
    try:
        logger.info("Début de l'explication LIME")
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image)
        logger.info(f"Image convertie en numpy array, shape: {img_np.shape}")
        
        # Calcul de l'embedding de référence
        with torch.no_grad():
            ref_tensor = transform(image).unsqueeze(0)
            ref_embedding = model(ref_tensor).squeeze().numpy()
            ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)
        
        def pred_fn(imgs):
            try:
                logger.info(f"Prédiction LIME en cours... batch size: {len(imgs)}")
                imgs_tensor = torch.stack([transform(Image.fromarray(img)) for img in imgs])
                logger.info(f"Tensors créés, shape: {imgs_tensor.shape}")
                with torch.no_grad():
                    embeddings = model(imgs_tensor).squeeze().numpy()
                    # Normalisation des embeddings
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / norms
                    
                    # Calcul de la similarité cosinus avec l'embedding de référence
                    similarities = np.dot(embeddings, ref_embedding)
                    
                    # On crée 3 classes pour mieux distinguer les caractéristiques
                    scores = np.zeros((len(embeddings), 3))
                    for i, sim in enumerate(similarities):
                        if sim > 0.8:  # Très similaire
                            scores[i, 0] = 1
                        elif sim > 0.5:  # Moyennement similaire
                            scores[i, 1] = 1
                        else:  # Peu similaire
                            scores[i, 2] = 1
                    
                logger.info(f"Scores générés, shape: {scores.shape}")
                return scores
            except Exception as e:
                logger.error(f"Erreur dans pred_fn: {str(e)}")
                raise
        
        logger.info("Création de l'explainer LIME")
        explainer = lime_image.LimeImageExplainer()
        logger.info("Génération de l'explication LIME avec 5000 échantillons")
        try:
            exp = explainer.explain_instance(
                img_np, 
                pred_fn, 
                num_samples=5000,  # Augmenté à 5000 échantillons
                top_labels=3,
                hide_color=0,
                batch_size=20,
                random_seed=42
            )
            logger.info("Explication LIME générée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors de explain_instance: {str(e)}")
            raise

        logger.info("Récupération de la heatmap LIME")
        # On utilise le label 0 qui correspond aux caractéristiques très distinctives
        heatmap, mask = exp.get_image_and_mask(
            0, 
            positive_only=True, 
            num_features=10,
            hide_rest=False
        )
        logger.info(f"Heatmap générée, shape: {heatmap.shape}")
        
        # Redimensionnement à la taille cible
        target_size = (224, 224)
        heatmap = np.array(Image.fromarray(heatmap).resize(target_size, Image.Resampling.LANCZOS))
        original = np.array(image.resize(target_size, Image.Resampling.LANCZOS))
        
        # Normalisation de la heatmap (en s'assurant qu'elle est en 2D)
        if len(heatmap.shape) == 3:
            heatmap = heatmap.mean(axis=2)  # Conversion en 2D si nécessaire
        
        # Normalisation plus fine avec percentile pour réduire l'effet des valeurs extrêmes
        p95 = np.percentile(heatmap, 95)
        heatmap = np.minimum(heatmap, p95)
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Création d'une heatmap colorée avec plus de nuances
        heatmap_colored = np.zeros((*heatmap_normalized.shape, 3), dtype=np.uint8)
        
        # Découpage plus fin des niveaux d'importance
        # Jaune clair pour les zones peu importantes
        mask_yellow_light = heatmap_normalized < 0.2
        heatmap_colored[mask_yellow_light] = [255, 255, 128]
        
        # Jaune pour les zones moyennement peu importantes
        mask_yellow = (heatmap_normalized >= 0.2) & (heatmap_normalized < 0.4)
        heatmap_colored[mask_yellow] = [255, 255, 0]
        
        # Orange clair pour les zones moyennement importantes
        mask_orange_light = (heatmap_normalized >= 0.4) & (heatmap_normalized < 0.6)
        heatmap_colored[mask_orange_light] = [255, 200, 0]
        
        # Orange pour les zones importantes
        mask_orange = (heatmap_normalized >= 0.6) & (heatmap_normalized < 0.8)
        heatmap_colored[mask_orange] = [255, 165, 0]
        
        # Rouge pour les zones très importantes
        mask_red = heatmap_normalized >= 0.8
        heatmap_colored[mask_red] = [255, 0, 0]
        
        # Superposition avec l'image originale
        alpha = 0.75  # Légèrement plus opaque pour mieux voir les détails
        overlay = (alpha * heatmap_colored + (1 - alpha) * original).astype(np.uint8)
        
        # Ajout d'un contour plus fin pour les zones très importantes
        edges = feature.canny(heatmap_normalized, sigma=1.5)  # Réduit pour des contours plus fins
        overlay[edges] = [255, 255, 255]  # Contour blanc
        
        # Création de l'image finale
        final_img = Image.fromarray(overlay)
        
        # Ajout d'une légende plus détaillée
        legend = Image.new('RGB', (224, 40), (255, 255, 255))
        draw = ImageDraw.Draw(legend)
        # Barre de couleur avec plus de nuances
        for i in range(224):
            if i < 45:
                color = (255, 255, 128)  # Jaune clair
            elif i < 90:
                color = (255, 255, 0)    # Jaune
            elif i < 135:
                color = (255, 200, 0)    # Orange clair
            elif i < 180:
                color = (255, 165, 0)    # Orange
            else:
                color = (255, 0, 0)      # Rouge
            draw.line([(i, 0), (i, 20)], fill=color)
        
        # Texte plus détaillé
        draw.text((5, 22), "Très peu", fill=(0, 0, 0))
        draw.text((45, 22), "Peu", fill=(0, 0, 0))
        draw.text((90, 22), "Moyen", fill=(0, 0, 0))
        draw.text((135, 22), "Important", fill=(0, 0, 0))
        draw.text((180, 22), "Très important", fill=(0, 0, 0))
        
        # Combinaison de l'image et de la légende
        final_with_legend = Image.new('RGB', (224, 264))
        final_with_legend.paste(final_img, (0, 0))
        final_with_legend.paste(legend, (0, 224))
        
        # Sauvegarde de l'image
        buffer = BytesIO()
        final_with_legend.save(buffer, format="PNG")
        logger.info("Explication LIME terminée avec succès")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Erreur détaillée dans compute_lime_explanation: {str(e)}\nType d'erreur: {type(e)}")
        raise

def compute_activation_map(image: Image.Image, model, transform) -> str:
    try:
        logger.info("Début du calcul de la carte d'activation")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        logger.info("Image transformée en tensor")
        
        with torch.no_grad():
            # Passer l'image à travers les couches du modèle jusqu'à layer2 (index 5)
            features = model[:6](image_tensor) 
            
            # On prend la moyenne sur les canaux pour obtenir la carte d'activation 2D
            activation = features.squeeze(0).mean(0).cpu().numpy()
        
        logger.info("Normalisation de la carte d'activation")
        # Suppression des valeurs négatives
        activation = np.maximum(activation, 0)
        
        # Normalisation min-max pour étirer les valeurs sur [0, 1]
        if activation.max() > activation.min(): 
            activation = (activation - activation.min()) / (activation.max() - activation.min())
        else:
            # Si toutes les activations sont les mêmes, remplir avec 0.5 (pour une échelle de gris neutre)
            activation = np.full_like(activation, 0.5)
        
        # Redimensionnement de la carte d'activation à la taille de l'image originale (en float)
        activation_resized = resize(activation, (224, 224), anti_aliasing=False) # Désactivation de l'anti-aliasing
        
        logger.info("Création de la heatmap en niveaux de gris")
        # Convertir directement en image grayscale (0-255)
        grayscale_heatmap = (activation_resized * 255).astype(np.uint8)
        
        # Superposition sur l'image originale
        original = np.array(image.resize((224, 224), Image.Resampling.LANCZOS))
        alpha = 0.7  # Opacité de la heatmap
        
        # Convertir l'image originale en niveaux de gris si nécessaire pour une superposition plus simple
        original_grayscale = Image.fromarray(original).convert("L")
        original_grayscale_np = np.array(original_grayscale)
        
        # Superposer la heatmap grayscale sur l'image originale grayscale
        overlay = (alpha * grayscale_heatmap + (1 - alpha) * original_grayscale_np).astype(np.uint8)
        final_img = Image.fromarray(overlay, mode="L") # Mode L pour niveaux de gris
        
        buffer = BytesIO()
        final_img.save(buffer, format="PNG")
        logger.info("Carte d'activation terminée avec succès")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Erreur détaillée dans compute_activation_map: {str(e)}")
        raise

@app.post("/embed", response_model=EmbeddingResponse)
async def get_image_embedding(image_url: ImageUrl):
    try:
        logger.info(f"Traitement de l'image: {image_url.url}")
        image = get_image_from_url(image_url.url)
        embedding = get_embedding(image)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lime", response_model=LimeExplanationResponse)
async def lime_endpoint(image_url: ImageUrl):
    try:
        logger.info(f"Début du traitement LIME pour l'image: {image_url.url}")
        image = get_image_from_url(image_url.url)
        logger.info("Image téléchargée avec succès")
        lime_b64 = compute_lime_explanation(image, model, transform)
        logger.info("Réponse LIME prête à être envoyée")
        return LimeExplanationResponse(lime_explanation=lime_b64)
    except Exception as e:
        logger.error(f"Erreur dans lime_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/activation-map", response_model=ActivationMapResponse)
async def activation_map_endpoint(image_url: ImageUrl):
    try:
        logger.info(f"Début du traitement activation map pour l'image: {image_url.url}")
        image = get_image_from_url(image_url.url)
        logger.info("Image téléchargée avec succès")
        activation_b64 = compute_activation_map(image, model, transform)
        logger.info("Réponse activation map prête à être envoyée")
        return ActivationMapResponse(activation_map=activation_b64)
    except Exception as e:
        logger.error(f"Erreur dans activation_map_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 