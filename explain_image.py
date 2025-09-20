import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import sys
import pandas as pd
from lime import lime_image
from skimage.segmentation import mark_boundaries
import cv2
import kagglehub
import matplotlib.pyplot as plt


def get_prediction(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities.cpu().numpy().flatten()


def get_lime_explanation(explainer, image_numpy, model_predict_fn, top_predictions, class_names):
    for i, (label_index, prob) in enumerate(top_predictions):
        explanation = explainer.explain_instance(
            image_numpy,
            model_predict_fn,
            top_labels=len(class_names),
            hide_color=0,
            num_samples=1000,
            segmentation_fn=None
        )
        temp, mask = explanation.get_image_and_mask(
            label_index,
            positive_only=False,
            num_features=5,
            hide_rest=False
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ax.set_title(f'LIME Explanation for Predicted Class: {class_names[label_index]}')
        ax.axis('off')
        plt.show()


def get_gradcam_explanation(model, image_tensor, top_predictions, class_names, device):
    for i, (label_index, prob) in enumerate(top_predictions):
        # We need to explicitly access the last convolutional layer
        final_conv_layer = model.layer4[-1]

        # Capture activations and gradients using hooks
        activation = {}
        gradient = {}

        def get_activation_hook(module, input, output):
            activation['final_conv'] = output.detach()

        def get_gradient_hook(module, grad_input, grad_output):
            gradient['final_conv'] = grad_output[0].detach()

        # Register the hooks
        activation_hook = final_conv_layer.register_forward_hook(get_activation_hook)
        gradient_hook = final_conv_layer.register_full_backward_hook(get_gradient_hook)

        # Forward pass to get activations
        image_tensor.requires_grad_(True)
        output = model(image_tensor.unsqueeze(0).to(device))

        # Get the predicted class score
        target_output = output[0, label_index]

        # Backward pass to get gradients
        model.zero_grad()
        target_output.backward(retain_graph=True)

        # Unregister the hooks
        activation_hook.remove()
        gradient_hook.remove()

        # Generate the heatmap
        # Get the activations and gradients
        activations = activation['final_conv']
        gradients = gradient['final_conv']

        # Pool the gradients to get weights for each activation channel
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Weighted sum of activations
        for k in range(activations.shape[1]):
            activations[:, k, :, :] *= pooled_gradients[k]

        # Create the heatmap
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        img = image_tensor.squeeze().detach().permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = np.float32(img) * 0.4 + np.float32(heatmap) * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255)
        superimposed_img = superimposed_img.astype(np.uint8)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(superimposed_img)
        ax.set_title(f'Grad-CAM Heatmap for Predicted Class: {class_names[label_index]}')
        ax.axis('off')
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python explain_image.py <path_to_image>")
        sys.exit(1)

    custom_image_path = sys.argv[1]
    model_filename = 'chest_xray_model.pth'

    if not os.path.exists(model_filename):
        print(f"Error: Model file '{model_filename}' not found.")
        print("Please run the training script ('xai_explanation_2.py') first to generate this file.")
        sys.exit(1)

    print(f"Loading pre-trained model from {model_filename}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Downloading the NIH Chest X-ray dataset from Kaggle to get class names...")
    path = kagglehub.dataset_download("nih-chest-xrays/data")
    data_dir = str(path)
    labels_df = pd.read_csv(os.path.join(data_dir, 'Data_Entry_2017.csv'))
    unique_labels = sorted(labels_df['Finding Labels'].unique())
    unique_single_labels = [label for label in unique_labels if '|' not in label]
    class_names = [label.replace('|', '/') for label in unique_single_labels]

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model.load_state_dict(torch.load(model_filename, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    try:
        image = Image.open(custom_image_path).convert('RGB')
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = data_transforms(image).to(device)
        image_numpy = image_tensor.permute(1, 2, 0).cpu().numpy()

    except FileNotFoundError:
        print(f"Error: Image file not found at '{custom_image_path}'.")
        sys.exit(1)

    predictions = get_prediction(model, image_tensor.cpu(), device)
    top_3_preds_indices = np.argsort(predictions)[::-1][:3]
    top_3_probs = predictions[top_3_preds_indices]
    top_predictions = list(zip(top_3_preds_indices, top_3_probs))

    print("\n--- Model Prediction and Explanations ---")
    for label_index, prob in top_predictions:
        print(f"Predicted Class: {class_names[label_index]}, Probability: {prob:.4f}")

    print("\nGenerating LIME explanations...")
    explainer = lime_image.LimeImageExplainer()


    def model_predict_fn(images):
        images_tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            outputs = model(images_tensor)
        return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()


    get_lime_explanation(explainer, image_numpy, model_predict_fn, top_predictions, class_names)

    print("\nGenerating Grad-CAM heatmaps...")
    get_gradcam_explanation(model, image_tensor, top_predictions, class_names, device)


