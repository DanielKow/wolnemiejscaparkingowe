import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

from saving_results import ResultsSaver


class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Cannot load the image. Check the file path.")
        self.result = self.image.copy()
        self.saver = ResultsSaver()
        self._save("original_image")

    def _save(self, step_name, image=None):
        """Internal method to save the current state of the image."""
        if image is None:
            image = self.result
        self.saver.save(image, step_name)

    def convert_to_grayscale(self):
        """Converts the current image to grayscale and saves the step."""
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)
        self._save("grayscale_image")
        return self

    def apply_bilateral_filter_bottom(self, d=9, sigma_color=75, sigma_space=75, mask_ratio=0.3):
        """Applies bilateral filtering only to the bottom portion of the image."""
        # Get the image dimensions
        height, width = self.result.shape[:2]
    
        # Define the region of interest (bottom 30% of the image)
        start_row = int(height * (1 - mask_ratio))
        roi = self.result[start_row:, :]
    
        # Apply bilateral filter to the bottom portion
        filtered_roi = cv2.bilateralFilter(roi, d, sigma_color, sigma_space)
    
        # Replace the bottom part of the original image with the filtered result
        self.result[start_row:, :] = filtered_roi
        self._save("bilateral_filter_bottom")
        return self

    def apply_kmeans_to_bottom(self, mask_ratio=0.3, k=5):
        """Applies K-means clustering to the bottom portion of the image and saves the step."""
        height, width = self.result.shape[:2]

        # Create a mask for the bottom region
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (0, int(height * (1 - mask_ratio))), (width, height), 255, -1)

        # Extract the bottom region
        bottom_region = cv2.bitwise_and(self.result, self.result, mask=mask)

        # Reshape the bottom region for K-means clustering
        pixel_values = bottom_region.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Map pixels to their cluster centers
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented_image = segmented.reshape(self.result.shape)

        # Combine the segmented bottom with the original top
        top_region = cv2.bitwise_and(self.result, self.result, mask=cv2.bitwise_not(mask))
        self.result = cv2.add(top_region, segmented_image)
        self._save("kmeans_segmented")
        return self

    def apply_clahe(self, clip_limit=2.0, grid_size=(8, 8)):
        """Enhances contrast using CLAHE and saves the step."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.result = clahe.apply(self.result)
        self._save("clahe_enhanced")
        return self

    def apply_gaussian_blur(self, kernel_size=(3, 3)):
        """Applies Gaussian blur and saves the step."""
        self.result = cv2.GaussianBlur(self.result, kernel_size, 0)
        self._save("blurred")
        return self
    
    
    def apply_gaussian_blur_bottom(self, kernel_size=(3, 3), mask_ratio=0.3):
        """Applies Gaussian blur only to the bottom portion of the image."""
        # Get the image dimensions
        height, width = self.result.shape[:2]
    
        # Define the region of interest (bottom 30% of the image)
        start_row = int(height * (1 - mask_ratio))
        roi = self.result[start_row:, :]
    
        # Apply Gaussian blur to the bottom portion
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
    
        # Replace the bottom part of the original image with the blurred result
        self.result[start_row:, :] = blurred_roi
        self._save("blurred_bottom")
        return self

    def detect_edges(self, threshold1=50, threshold2=150):
        """Performs edge detection using the Canny algorithm and saves the step."""
        self.result = cv2.Canny(self.result, threshold1, threshold2)
        self._save("edges_detected")
        return self

    def refine_edges(self, kernel_size=(2, 2), iterations=1):
        """Applies morphological operations to refine edges and saves the step."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.result = cv2.dilate(self.result, kernel, iterations=iterations)
        self.result = cv2.erode(self.result, kernel, iterations=iterations)
        self._save("refined_edges")
        return self

    def detect_lines_hough(self, min_line_length=80, max_line_gap=20, threshold=100):
        """Detects lines using the Hough Line Transform and saves the step."""
        lines = cv2.HoughLinesP(self.result, 1, np.pi / 180, threshold, minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
        line_image = np.copy(self.image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    slope = abs((y2 - y1) / (x2 - x1))
                    if slope > 5:  # Vertical lines only
                        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.result = line_image
        self._save("final_detected_lines")
        return self

    def detect_lines_lsd(self):
        """Detects parking slots based on the detected lines and saves the step."""
        lsd = cv2.createLineSegmentDetector()

        # Detect lines
        lines = lsd.detect(self.result)[0]
        
        # Create a copy of the original image for drawing
        lines_image = self.image.copy()
        
        # Draw the detected lines
        if lines is None:
            print("No lines detected.")
            return self
        
        for line in lines:
            x1, y1, x2, y2 = line.flatten().astype(int)
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display the result
        self.result = lines_image
        self._save("vertical_lines")
        return self

    def detect_lines_ransac(self):
        """Detects lines using the RANSAC algorithm and saves the step."""
        # Extract the edge points from the image
        y_coords, x_coords = np.where(self.result > 0)  # Edge pixels (non-zero)

        if len(x_coords) < 2:  # Not enough points to fit a line
            print("Not enough edge points for RANSAC line detection.")
            return self

        # Stack the coordinates
        points = np.column_stack((x_coords, y_coords))

        # Initialize RANSAC
        ransac = RANSACRegressor(residual_threshold=10, max_trials=100)

        # Fit RANSAC to the points
        ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

        # Predict the inlier line
        line_x = np.linspace(0, self.result.shape[1], 1000)  # x-coordinates for line
        line_y = ransac.predict(line_x.reshape(-1, 1))  # Corresponding y-coordinates

        # Convert to integer for drawing
        line_x = line_x.astype(int)
        line_y = line_y.astype(int)

        # Create a copy of the original image for drawing
        line_image = self.image.copy()

        # Draw the RANSAC line
        for i in range(len(line_x) - 1):
            if 0 <= line_x[i] < line_image.shape[1] and 0 <= line_y[i] < line_image.shape[0]:
                cv2.line(line_image, (line_x[i], line_y[i]), (line_x[i + 1], line_y[i + 1]), (0, 0, 255), 2)

        # Save and display the result
        self.result = line_image
        self._save("ransac_detected_lines")
        return self

    def mark_free_spaces(self):
        mask1 = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
        mask2 = self.result
        
        _, mask1_binary = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        _, mask2_binary = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask1_binary, connectivity=8)
        
        result_mask = mask1_binary.copy()
        
        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8) * 255
        
            intersection = cv2.bitwise_and(component_mask, mask2_binary)
            if cv2.countNonZero(intersection) > 0:
                result_mask[labels == label] = 0

        green_overlay = np.zeros_like(self.image, dtype=np.uint8)
        green_overlay[:] = (0, 255, 0)
        
        masked_green = cv2.bitwise_and(green_overlay, green_overlay, mask=result_mask)
        alpha = 0.5
        result = cv2.addWeighted(masked_green, alpha, self.image, 1 - alpha, 0)
        
        self.result = result
        self._save("free_spaces")
        return self


    def draw_histogram(self):
        """Draws a histogram of pixel intensities and saves the step."""
        hist = cv2.calcHist([self.result], [0], None, [256], [0, 256])
        plt.figure(figsize=(10, 5))
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
        plt.close()
        return self

    def display_results(self):
        """Displays all saved images."""
        self.saver.display_images()


if __name__ == "__main__":
    processor = ImageProcessor('test_images/1.jpg')
    
    processor \
        .convert_to_grayscale() \
        .apply_gaussian_blur() \
        .apply_gaussian_blur_bottom(kernel_size=(5, 5)) \
        .draw_histogram()  \
        .apply_kmeans_to_bottom(k=2) \
        .draw_histogram() \
        .apply_bilateral_filter_bottom(sigma_color=90, sigma_space=90) \
        .draw_histogram() \
        .apply_clahe() \
        .draw_histogram() \
        .detect_edges() \
        .refine_edges() \
        .mark_free_spaces() \
        .display_results()
