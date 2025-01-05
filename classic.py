import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

from saving_results import ResultsSaver


class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Cannot load the image. Check the file path.")
        self.result = self.image.copy()  
        self.saver = ResultsSaver()
        self._save("original_image")

    def _save(self, step_name):
        """Internal method to save the current state of the image."""
        self.saver.save(self.result, step_name)

    def convert_to_grayscale(self):
        """Converts the current image to grayscale and saves the step."""
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)
        self._save("grayscale_image")
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

    def apply_blur(self, kernel_size=(5, 5)):
        """Applies Gaussian blur and saves the step."""
        self.result = cv2.GaussianBlur(self.result, kernel_size, 0)
        self._save("blurred")
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
        lines = cv2.HoughLinesP(self.result, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
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

        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 5:  # Vertical line threshold (small horizontal difference)
                vertical_lines.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))

        # Draw the detected lines
        drawn_lines = self.image.copy()
        for line in vertical_lines:
            x1, y1, x2, y2 = line
            cv2.line(drawn_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the result
        self.result = drawn_lines
        self._save("detected_parking_slots")
        
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
        line_y = ransac.predict(line_x.reshape(-1, 1))       # Corresponding y-coordinates
    
        # Convert to integer for drawing
        line_x = line_x.astype(int)
        line_y = line_y.astype(int)
    
        # Create a copy of the original image for drawing
        line_image = self.image.copy()
    
        # Draw the RANSAC line
        for i in range(len(line_x) - 1):
            if 0 <= line_x[i] < line_image.shape[1] and 0 <= line_y[i] < line_image.shape[0]:
                cv2.line(line_image, (line_x[i], line_y[i]), (line_x[i+1], line_y[i+1]), (0, 0, 255), 2)
    
        # Save and display the result
        self.result = line_image
        self._save("ransac_detected_lines")
        return self

    def display_results(self):
        """Displays all saved images."""
        self.saver.display_images()


processor = ImageProcessor('test_images/2012-09-11_16_48_36_jpg.rf.4ecc8c87c61680ccc73edc218a2c8d7d.jpg')

processor.convert_to_grayscale() \
    .apply_kmeans_to_bottom(mask_ratio=0.3, k=5) \
    .apply_clahe() \
    .apply_blur(kernel_size=(3,3)) \
    .detect_edges() \
    .refine_edges() \
    .detect_lines_lsd() \
    .display_results()
