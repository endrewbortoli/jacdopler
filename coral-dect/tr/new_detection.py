import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar as imagens
img_pov_path = "/img/pic5.png"
img_field_path = "/2025.png"

img_pov = cv2.imread(img_pov_path)
img_field = cv2.imread(img_field_path)

# Converter a imagem para RGB (matplotlib usa RGB, enquanto OpenCV usa BGR)
img_pov_rgb = cv2.cvtColor(img_pov, cv2.COLOR_BGR2RGB)
img_field_rgb = cv2.cvtColor(img_field, cv2.COLOR_BGR2RGB)

# Mostrar as imagens para referência
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(img_pov_rgb)
axes[0].set_title("Imagem POV da Arena")
axes[0].axis("off")

axes[1].imshow(img_field_rgb)
axes[1].set_title("Imagem de Cima da Arena")
axes[1].axis("off")

plt.show()
