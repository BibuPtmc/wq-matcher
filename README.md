# Service d'Embedding d'Images

Ce microservice utilise ResNet18 pour générer des embeddings d'images à partir d'URLs Cloudinary.

## Prérequis

- Python 3.8+
- pip

## Installation

1. Créer un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Démarrage

Pour démarrer le service :

```bash
python main.py
```

Le service sera accessible à l'adresse : `http://localhost:8000`

## Endpoints

### POST /embed

Génère un embedding pour une image à partir de son URL.

**Request Body :**

```json
{
  "url": "https://res.cloudinary.com/example/image/upload/v1234567890/example.jpg"
}
```

**Response :**

```json
{
    "embedding": [0.1, 0.2, 0.3, ...]  // Vecteur de 512 dimensions
}
```

### GET /health

Vérifie l'état du service.

**Response :**

```json
{
  "status": "healthy"
}
```

## Documentation API

La documentation Swagger est disponible à l'adresse : `http://localhost:8000/docs`

fastapi==0.104.1
C’est le framework principal de ton API.
Il te permet de créer rapidement des endpoints REST (comme /embed) avec de la validation automatique via Pydantic, de la documentation Swagger générée automatiquement, etc.

uvicorn==0.24.0
C’est le serveur ASGI qui exécute ton app FastAPI.
Il est rapide et asynchrone, idéal pour les APIs modernes. Il lance l’app dans if **name** == "**main**".

Upload et parsing de fichiers

python-multipart==0.0.6
Nécessaire si tu veux gérer des fichiers envoyés par formulaire dans FastAPI (Form, File, UploadFile).
Tu n’en as pas encore besoin ici (car tu envoies des URLs), mais utile si un jour tu veux envoyer l’image directement.

Modèle de deep learning
torch==2.7.0
PyTorch, le framework de deep learning.

Tu l’utilises pour charger le modèle ResNet18, faire des prédictions, manipuler les tenseurs, etc.

torchvision==0.22.0
Utilisé pour :
Charger les modèles pré-entraînés (resnet18),
Appliquer les transformations d’image (transforms).
Complément indispensable à torch pour les projets de vision par ordinateur.

Pillow==10.1.0
Librairie pour manipuler des images (ouvrir, redimensionner, convertir).

Image.open() vient de là.

numpy==1.26.2
Librairie mathématique.

Utilisée ici pour normaliser les embeddings : embedding / np.linalg.norm(embedding)

requests==2.31.0
Pour faire des appels HTTP.

Tu l’utilises pour télécharger l’image depuis l’URL envoyée par l’utilisateur.
