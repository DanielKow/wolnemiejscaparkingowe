import os
import matplotlib.pyplot as plt
import cv2

class ResultsSaver:
    def __init__(self):
        results_path = "results"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    
        # Determine the next run number
        existing_runs = [d for d in os.listdir(results_path) if d.startswith("run_") and d[4:].isdigit()]
        next_run_number = 1 + max([int(d[4:]) for d in existing_runs], default=0)
    
        # Create the next run directory
        self.run_dir = os.path.join(results_path, f"run_{next_run_number}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.images_and_titles = []

    def save(self, image, title):
        """
        Saves the given image to the specified run directory.
        """
        filepath = os.path.join(self.run_dir, title + ".jpg")
        
        cv2.imwrite(filepath, image)
        print(f"Saved: {filepath}")
        self.images_and_titles.append((image, title))

    def display_images(self):
        """Displays all saved images dynamically."""
    
        num_images = len(self.images_and_titles)  # Total number of images
        cols = 4  # Number of columns (can be adjusted)
        rows = (num_images + cols - 1) // cols  # Calculate rows dynamically
    
        plt.figure(figsize=(15, 5 * rows))
        for i, (image, title) in enumerate(self.images_and_titles):
            plt.subplot(rows, cols, i + 1)
            if len(image.shape) == 2:  # Grayscale
                plt.imshow(image, cmap='gray')
            else:  # Color
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()