import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image loading function
def load_image(img_path, shape=None):
    image = Image.open(img_path).convert('RGB')
    in_transform = transforms.Compose([
        transforms.Resize(shape) if shape else transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

# Image unloader
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

# Image save function
def save_image(tensor, path):
    img = im_convert(tensor)
    img = Image.fromarray((img * 255).astype('uint8'))
    img.save(path)

# Load images
content = load_image(r"C:\codtech_internship\my codes\1_ai\content_image.jpg")
style = load_image(r"C:\codtech_internship\my codes\1_ai\style_image.jpg", shape=content.shape[-2:])

# Load pretrained VGG19
from torchvision.models import vgg19, VGG19_Weights
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# Feature extractor
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

# Extract features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Target image
target = content.clone().requires_grad_(True).to(device)

# Weights
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Training
steps = 500
show_every = 100
save_dir = r"C:\codtech_internship\my codes"

for i in range(1, steps + 1):
    print(f"Running step {i}...")

    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_loss / (target_feature.shape[1] ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % show_every == 0:
        print(f"Step {i}, Total Loss: {total_loss.item():.4f}")
        image_path = os.path.join(save_dir, f"output_step_{i}.jpg")
        save_image(target, image_path)
        print(f"Saved intermediate image: {image_path}")

# Save final image
final_path = os.path.join(save_dir, "stylized_output.jpg")
save_image(target, final_path)
print(f"Final stylized image saved as: {final_path}")
