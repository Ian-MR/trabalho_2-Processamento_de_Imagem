import cv2
import numpy as np
import matplotlib.pyplot as plt

# Função para exibir imagens
def show_image(image, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Carregar a imagem
image = cv2.imread('foto.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem")
    exit()

# Aplica a Transformada de Fourier
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# Desloca a componente de baixa frequência para o centro da imagem
dft_shift = np.fft.fftshift(dft)

# Magnitude do espectro
magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

# Exibir a magnitude do espectro
show_image(np.log(magnitude_spectrum + 1), "Magnitude Spectrum")

# Criar o filtro passa-baixa (filtro gaussiano)
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# Filtro Gaussiano
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), 50, (1, 1, 1), -1)

#mask = np.ones((rows, cols, 2), np.uint8)
#cv2.circle(mask, (ccol, crow), 50, (0, 0, 0), -1)

# Aplica o filtro passa-baixa
fshift = dft_shift * mask

# Exibir o espectro filtrado
magnitude_spectrum_filtered = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])
show_image(np.log(magnitude_spectrum_filtered + 1), "Filtered Spectrum")

# Reverter a transformada de Fourier
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Exibir a imagem resultante
show_image(img_back, "Imagem Reconstruída")

# Salvar a imagem reconstruída
cv2.imwrite('imagem_reconstruida.jpg', img_back)
