import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import os
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
        if image is None:
            image = self.result
        self.saver.save(image, step_name)

    def convert_to_grayscale(self):
        self.result = cv2.cvtColor(self.result, cv2.COLOR_BGR2GRAY)
        self._save("grayscale_image")
        return self

    def apply_bilateral_filter_bottom(self, d=9, sigma_color=75, sigma_space=75, mask_ratio=0.3):
        height, width = self.result.shape[:2]
    
        start_row = int(height * (1 - mask_ratio))
        roi = self.result[start_row:, :]
    
        filtered_roi = cv2.bilateralFilter(roi, d, sigma_color, sigma_space)
    
        # Replace the bottom part of the original image with the filtered result
        self.result[start_row:, :] = filtered_roi
        self._save("bilateral_filter_bottom_d={}_color={}_space={}_mask={}".format(d, sigma_color, sigma_space, mask_ratio))
        return self

    def apply_adaptive_thresholding(self, max_value=255, block_size=11, constant=2):
        self.result = cv2.adaptiveThreshold(self.result, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, block_size, constant)
        self._save("adaptive_thresholded_max={}_block={}_constant={}".format(max_value, block_size, constant))
        return self

    def apply_kmeans_to_bottom(self, mask_ratio=0.3, k=5):
        height, width = self.result.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (0, int(height * (1 - mask_ratio))), (width, height), 255, -1)

        bottom_region = cv2.bitwise_and(self.result, self.result, mask=mask)

        pixel_values = bottom_region.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()]
        segmented_image = segmented.reshape(self.result.shape)

        top_region = cv2.bitwise_and(self.result, self.result, mask=cv2.bitwise_not(mask))
        self.result = cv2.add(top_region, segmented_image)
        self._save("kmeans_segmented_mask={}_k={}".format(mask_ratio, k))
        return self

    def apply_clahe(self, clip_limit=2.0, grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.result = clahe.apply(self.result)
        self._save("clahe_enhanced_clip={}_grid={}".format(clip_limit, grid_size))
        return self

    def apply_gaussian_blur(self, kernel_size=(3, 3)):
        self.result = cv2.GaussianBlur(self.result, kernel_size, 0)
        self._save("blurred_kernel={}".format(kernel_size))
        return self
    
    
    def apply_gaussian_blur_bottom(self, kernel_size=(3, 3), mask_ratio=0.3):
        height, width = self.result.shape[:2]
    
        start_row = int(height * (1 - mask_ratio))
        roi = self.result[start_row:, :]
    
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
    
        self.result[start_row:, :] = blurred_roi
        self._save("blurred_bottom_kernel={}_mask={}".format(kernel_size, mask_ratio))
        return self

    def detect_edges(self, threshold1=50, threshold2=150):
        self.result = cv2.Canny(self.result, threshold1, threshold2)
        self._save("edges_detected_threshold1={}_threshold2={}".format(threshold1, threshold2))
        return self

    def refine_edges(self, kernel_size=(2, 2), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.result = cv2.dilate(self.result, kernel, iterations=iterations)
        self.result = cv2.erode(self.result, kernel, iterations=iterations)
        self._save("refined_edges_kernel={}_iterations={}".format(kernel_size, iterations))
        return self

    def detect_lines_hough(self, min_line_length=80, max_line_gap=20, threshold=100):
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
        self._save("final_detected_lines_hough_min={}_max={}_threshold={}".format(min_line_length, max_line_gap, threshold))
        return self

    def detect_lines_lsd(self):
        lsd = cv2.createLineSegmentDetector()

        lines = lsd.detect(self.result)[0]
        
        lines_image = self.image.copy()
        
        if lines is None:
            print("No lines detected.")
            return self
        
        for line in lines:
            x1, y1, x2, y2 = line.flatten().astype(int)
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        self.result = lines_image
        self._save("vertical_lines")
        return self
    
    def apply_opened_morphology(self, kernel_size=(3, 3), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.result = cv2.morphologyEx(self.result, cv2.MORPH_OPEN, kernel, iterations=iterations)
        self._save("opened_morphology_kernel={}_iterations={}".format(kernel_size, iterations))
        return self
    
    def apply_closed_morphology(self, kernel_size=(3, 3), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.result = cv2.morphologyEx(self.result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        self._save("closed_morphology_kernel={}_iterations={}".format(kernel_size, iterations))
        return self

    def apply_erode_morphology(self, kernel_size=(3, 3), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.result = cv2.erode(self.result, kernel, iterations=iterations)
        self._save("eroded_morphology_kernel={}_iterations={}".format(kernel_size, iterations))
        return self
    
    def apply_dilate_morphology(self, kernel_size=(3, 3), iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.result = cv2.dilate(self.result, kernel, iterations=iterations)
        self._save("dilated_morphology_kernel={}_iterations={}".format(kernel_size, iterations))
        return self
    
    def detect_lines_ransac(self):
        y_coords, x_coords = np.where(self.result > 0)  # Edge pixels (non-zero)

        if len(x_coords) < 2:  # Not enough points to fit a line
            print("Not enough edge points for RANSAC line detection.")
            return self

        points = np.column_stack((x_coords, y_coords))
        ransac = RANSACRegressor(residual_threshold=10, max_trials=100)
        ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])
        
        line_x = np.linspace(0, self.result.shape[1], 1000)  # x-coordinates for line
        line_y = ransac.predict(line_x.reshape(-1, 1))  # Corresponding y-coordinates

        line_x = line_x.astype(int)
        line_y = line_y.astype(int)

        line_image = self.image.copy()

        for i in range(len(line_x) - 1):
            if 0 <= line_x[i] < line_image.shape[1] and 0 <= line_y[i] < line_image.shape[0]:
                cv2.line(line_image, (line_x[i], line_y[i]), (line_x[i + 1], line_y[i + 1]), (0, 0, 255), 2)

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
        self.saver.display_images()
        return self
        
    def display_final_result(self):
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        return self


def process_image(image_path):
    processor = ImageProcessor(image_path)

    processor \
        .convert_to_grayscale() \
        .apply_gaussian_blur(kernel_size=(13,13)) \
        .apply_gaussian_blur_bottom() \
        .apply_bilateral_filter_bottom(mask_ratio=1) \
        .apply_kmeans_to_bottom(k=12, mask_ratio=1) \
        .detect_edges() \
        .refine_edges() \
        .mark_free_spaces() \
        .display_results() \
        .display_final_result()

def get_images_to_process():
    image_files = []

    for filename in os.listdir("test_images"):
        if filename.lower().endswith(".jpg"):  # Check valid image extension
            image_files.append(os.path.join("test_images", filename))
            
    return image_files

if __name__ == "__main__":
    images = get_images_to_process()
    for image in images[3:5]:
        process_image(image)
        print(f"Przetworzono: {image}")
        print("=" * 50)
        
