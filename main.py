from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import List
import logging

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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 