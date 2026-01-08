import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_void_content(image_path):
    img = cv2.imread(image_path, 0)
    # filter using Gaussian blur 
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Thresholding (The Fix)
    # Instead of Otsu (which captures weave shadows), we use a Manual Threshold.
    # We look for pixels darker than 50 (on a scale of 0-255).
    # 0 is black, 255 is white. Voids are near 0.
    manual_threshold_value = 50 
    _, mask = cv2.threshold(img_blurred, manual_threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Morphological Operations
    kernel = np.ones((3,3), np.uint8)
    
    # Image Opening (Erosion -> Dilation)
    # Removes tiny white specks (noise) that are too small to be real voids
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Image Closing (Dilation -> Erosion)
    # Fills in small holes inside the detected voids to make them solid shapes
    mask_cleaned = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)

    total_pixels = mask_cleaned.size
    void_pixels = np.count_nonzero(mask_cleaned)
    void_percentage = (void_pixels / total_pixels) * 100

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original Image
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    
    # Histogram
    # sum of 2 Gaussians 
    # You will likely see a large hill on the right (fabric) and a tiny tail on the left (voids).
    ax[1].hist(img.ravel(), 256, [0, 256], color='gray')
    ax[1].axvline(manual_threshold_value, color='r', linestyle='--', label=f'Threshold: {manual_threshold_value}')
    ax[1].set_title('Pixel Intensity Histogram')
    ax[1].legend()
    ax[2].imshow(mask_cleaned, cmap='gray')
    ax[2].set_title(f'Detected Voids (White)\nContent: {void_percentage:.2f}%')
    plt.tight_layout()
    plt.show()

    print(f"Calculated Void Content: {void_percentage:.2f}%")
    if void_percentage < 2.0:
        print(">>> CONCLUSION: The void content is ACCEPTABLE (less than 2%).")
    else:
        print(">>> CONCLUSION: The void content is UNACCEPTABLE (greater than 2%).")

if __name__ == "__main__":
    calculate_void_content('C:\\Users\\User\\OneDrive - Delft University of Technology\\Master Y1\\Polymer Composite Manufacturing\\Notebook 4 git\\PCM-notebooks\\slice_0130.jpeg')