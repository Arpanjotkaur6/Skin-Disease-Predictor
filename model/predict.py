import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent

# Load class names
with open(BASE_DIR / "utils" / "dataset_info.json", "r") as f:
    dataset_info = json.load(f)

NUM_CLASSES = dataset_info["num_classes"]
CLASS_NAMES = dataset_info["class_names"]
SEVERITY_LABELS = ["Mild", "Moderate", "Severe"]

class SkinDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(SkinDiseaseModel, self).__init__()
        self.backbone = models.efficientnet_b0(weights="DEFAULT")
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.disease_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        self.severity_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        features     = self.backbone(x)
        disease_out  = self.disease_head(features)
        severity_out = self.severity_head(features)
        return disease_out, severity_out

def load_model():
    model = SkinDiseaseModel(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(BASE_DIR / "checkpoints" / "best_model.pth",
                   map_location=device)
    )
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

def get_disease_key(disease_name):
    return disease_name.replace("human_", "")                       .replace("dog_", "")                       .replace("cat_", "")

def adjust_severity(severity_idx, patient_data):
    age       = patient_data.get("age", 25)
    duration  = patient_data.get("duration", "Less than 1 week")
    spreading = patient_data.get("spreading", False)
    itchy     = patient_data.get("itchy", False)
    species   = patient_data.get("species", "Human")
    if species == "Human":
        if age > 60 or age < 5:
            severity_idx = min(severity_idx + 1, 2)
    if "More than 1 month" in duration:
        severity_idx = min(severity_idx + 1, 2)
    elif "2-4 weeks" in duration:
        severity_idx = min(severity_idx + 1, 2)
    if spreading:
        severity_idx = min(severity_idx + 1, 2)
    if itchy and severity_idx == 0:
        severity_idx = 1
    return severity_idx

def predict(image_path, patient_data, model):
    original_image, image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        disease_out, severity_out = model(image_tensor)
    disease_probs = torch.softmax(disease_out, dim=1)
    disease_idx   = disease_probs.argmax().item()
    confidence    = float(disease_probs.max().item()) * 100
    top5_probs, top5_idx = torch.topk(disease_probs, 5)
    top5 = [
        (CLASS_NAMES[idx], float(prob) * 100)
        for idx, prob in zip(top5_idx[0], top5_probs[0])
    ]
    severity_idx = severity_out.argmax().item()
    severity_idx = adjust_severity(severity_idx, patient_data)
    severity     = SEVERITY_LABELS[severity_idx]
    disease_name = CLASS_NAMES[disease_idx]
    disease_key  = get_disease_key(disease_name)
    return {
        "disease":    disease_name.replace("_", " ").title(),
        "confidence": round(confidence, 2),
        "severity":   severity,
        "top5":       top5,
        "image":      original_image,
        "disease_key": disease_key
    }
