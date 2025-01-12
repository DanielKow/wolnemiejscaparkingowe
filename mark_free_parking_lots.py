import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_red_rectangles():
    image = cv2.imread("empty3.jpg")
    
    lower_red= np.array([0, 0, 249])
    upper_red = np.array([0, 0, 255])
    mask = cv2.inRange(image, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    return contours


rectangles = find_red_rectangles()

original_image = cv2.imread("empty.jpg")
image = original_image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

for rectangle in rectangles:
    points = np.array(rectangle, np.int32)  # Konwersja punktów do formatu NumPy
    points = points.reshape((-1, 1, 2))  # Dopasowanie wymiarów
    cv2.polylines(image, [points], isClosed=True, color=(255, 0, 0), thickness=3)  # Kolor czerwony (RGB)

# Wyświetlenie obrazu za pomocą matplotlib.pyplot
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')  # Ukrycie osi
plt.title("Obraz z narysowanymi czworokątami")
plt.show()