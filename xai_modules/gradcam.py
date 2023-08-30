# gradcam.py

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn as nn
from PIL import Image

def visualize_gradcam(model,
                      model_type: str,
                      image,
                      exp_target: int,
                      loss_fn,
                      final_layer,
                      img_size: tuple = (224, 224), 
                      device: torch.device = torch.device('cuda:0')):

  final_layer = final_layer
  activations = None

  transform1 = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    ])
  normalize_tf  = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]
                                     )
  
  if model_type == 'CNN':
    for m in final_layer.modules():
      if isinstance(m, nn.Conv2d):
          layer = m
          break
  elif model_type == 'ViT':
    for m in final_layer.modules():
        if isinstance(m, nn.LayerNorm):
            layer = m
            break
     
  def idx_to_onehot(idx):
    onehot = {0: [1,0,0], 1: [0,1,0], 2: [0,0,1]}
    return onehot[idx]

  def hook(module, input, output):
      nonlocal activations
      activations = output.detach()

  handle = final_layer.register_forward_hook(hook)

  transformed_img = transform1(image)
  input_img = normalize_tf(transformed_img).to(device).unsqueeze(0)

  
  if model_type == 'CNN':
    for param in model.features.parameters():
      param.requires_grad = True
  elif model_type == 'ViT':
    pass
  
  # Forward pass
  model.eval()
  output = model(input_img) 
  pred = int((torch.argmax(output, dim=-1)).item())

  # Remove hook
  handle.remove()

  target_label = torch.FloatTensor(idx_to_onehot(exp_target)).unsqueeze(0).to(device)

  # Compute loss
  loss = loss_fn(output, target_label) 
  loss.backward()

  grads = layer.weight.grad

 # Generate heatmap
  weights = torch.mean(grads, dim=[1,2]) # Average gradient for each filter
  cam = torch.matmul(activations.reshape([7*7,1280]), weights)
  cam = cam.reshape(7, 7) # 7x7 activation map
  cam = cam - torch.min(cam)
  cam = cam / torch.max(cam)
  heatmap = cv2.resize(cam.detach().cpu().numpy(), (224, 224))

  # Overlay heatmap on image
  img = transformed_img.detach().permute(1,2,0).cpu().numpy().squeeze()

  # Normalize the heatmap between 0 and 1
  heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

  # Resize the heatmap to match the size of the original image
  heatmap = np.uint8(255 * heatmap)
  heatmap = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]))

  # Convert the heatmap to a numpy array
  heatmap = np.array(heatmap)

  for param in model.features.parameters():
      param.requires_grad = False


  # Set color map
  cmap = plt.get_cmap('jet')

  # Model
  print(f'Predicted Class: {pred}')
  print(f'Expected Class: {exp_target}')

  # Model confidence 
  confidence = torch.max(model(input_img)).item()
  print(f"Model confidence: {confidence:.4f}")
  
  # Model sensitivity
  sensitivity = torch.mean(output).item()
  print(f"Model sensitivity: {sensitivity:.4f}" )

  # Display the result using matplotlib
  fig, axs = plt.subplots(1, 2, dpi = 150, figsize = (12, 5))
  axs[0].imshow(img)
  axs[0].set_title("Original Image")
  axs[0].axis('off')
  axs[1].imshow(img)
  axs[1].imshow(heatmap, cmap = cmap, alpha = 0.3)
  axs[1].set_title("Heatmap Overlay")
  axs[1].axis('off')

  # Add colorbar 
  colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap = cmap), ax=axs[1], fraction=0.046, pad=0.04)
  colorbar.set_label('Activation Pixels Impact Regions')

  plt.suptitle("GradCAM mappings")
  plt.tight_layout()
  plt.show()