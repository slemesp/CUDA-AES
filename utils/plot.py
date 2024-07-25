from matplotlib import pyplot as plt


def plot_interactive_image(image_data, encrypted_data, decrypted_data, crop_size=100):
    # Comprobar si la imagen tiene una dimensión adicional
    if image_data.shape[0] == 1:
        image_data = image_data[0]
        encrypted_data = encrypted_data[0]
        decrypted_data = decrypted_data[0]

    # Definir la región de interés (ROI) para recortar
    print(image_data.shape)
    height, width = image_data.shape
    start_row = (height - crop_size) // 2
    start_col = (width - crop_size) // 2

    # Recortar la región de 300x300
    image_crop = image_data[start_row:start_row + crop_size, start_col:start_col + crop_size]
    encrypted_crop = encrypted_data[start_row:start_row + crop_size, start_col:start_col + crop_size]
    decrypted_crop = decrypted_data[start_row:start_row + crop_size, start_col:start_col + crop_size]

    # Plotear las tres imágenes
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Imagen original
    axs[0].imshow(image_crop, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Imagen encriptada (puede no tener sentido visualizarla como una imagen, pero se muestra para completar el ejemplo)
    axs[1].imshow(encrypted_crop, cmap='gray')
    axs[1].set_title('Encrypted Image')
    axs[1].axis('off')

    # Imagen desencriptada
    axs[2].imshow(decrypted_crop, cmap='gray')
    axs[2].set_title('Decrypted Image')
    axs[2].axis('off')

    # Mostrar el plot
    plt.tight_layout()
    plt.show()
