import matplotlib.pyplot as plt

def show_images(images):
    for number, img in enumerate(images):
        plt.subplot(1, len(images), number+1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
    plt.show()