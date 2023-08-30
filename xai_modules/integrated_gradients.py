# integrated_grads.py
import torch
import numpy as np 
import matplotlib.pyplot as plt
from torchvision import transforms
from captum.attr import IntegratedGradients
import warnings
warnings.filterwarnings("ignore") 


def visualize_integrated_grads(image, 
                               model, 
                               exp_target,
                               grad_threshold,
                               ig_steps: int = 10,
                               img_size = (224, 224), 
                               device: torch.device = torch.device('cuda:0')):
    
    transform1 = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        ])
    
    normalize_tf  = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]
                                        )


    transformed_img = transform1(image)
    input_img = normalize_tf(transformed_img).to(device).unsqueeze(0)

    # Predicted Class:
    output = model(input_img.to(device))
    pred = int((torch.argmax(output, dim=-1)).item())

    # Create an IntegratedGradients object
    ig = IntegratedGradients(model)

    # Calculate the attributions
    attr = ig.attribute(input_img, target=pred, n_steps=ig_steps)

    image_np = np.transpose(transformed_img.cpu().detach().numpy(), (1, 2, 0))

    heatmap = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # Adjust this threshold to control the level of highlighting
    threshold = grad_threshold

    # Transfer the attr tensor to CPU and convert it to a NumPy array
    attr_np = attr.cpu().detach().numpy()

    # Apply threshold to the attributions
    thresholded_attr = np.where(attr_np >= threshold, attr_np, 0)

    heatmap_thres = np.transpose(thresholded_attr.squeeze(), (1, 2, 0))

    # Apply heatmap as an overlay on the image
    highlighted = np.where(heatmap_thres > 0, image_np + heatmap_thres, image_np)

    # Model 
    print(f'Predicted Class: {pred}')
    print(f'Expected Class: {exp_target}')

    # Model confidence
    confidence = torch.max(model(input_img)).item()
    print(f"Model confidence: {confidence:.4f}")
    
    # Model sensitivity
    sensitivity = torch.mean(output).item() 
    print(f"Model sensitivity: {sensitivity:.4f}" )

    # Integrated_Grad sensitivity
    ig_sensitivity = np.mean(attr_np)
    print(f"Integrated_Grad sensitivity: {ig_sensitivity:.4f}", )

    # Plot the overlayed image
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), dpi = 150)

    ax[0].imshow((image_np * 255).astype(np.uint8))
    ax[0].axis('off')
    ax[0].set_title("Original Image")
    ax[1].imshow(heatmap)
    ax[1].set_title("Generated Gradients")
    ax[1].axis('off')
    ax[2].imshow((image_np * 255).astype(np.uint8))
    ax[2].imshow(highlighted, alpha = 0.9)
    ax[2].set_title(f"Gradients Overlay on Image (Threshold: {threshold})")
    ax[2].axis('off')

    plt.suptitle("Integrated Gradients Explanation")
    plt.tight_layout()
    plt.show()