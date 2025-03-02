import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

def process_input_image(img_path):
    image_data = cv2.imread(img_path)
    if image_data is None:
        raise IOError("Image not found or could not be opened. Verify the path.")
    
    grayscale_img = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    
    adaptive_hist = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
    enhanced_gray = adaptive_hist.apply(grayscale_img)
    
    blurred_image = cv2.GaussianBlur(enhanced_gray, (19, 19), 0)
    
    return image_data, grayscale_img, enhanced_gray, blurred_image

def locate_coins_hough(blurred_img, original_img):
    detected_circles = cv2.HoughCircles(
        blurred_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=32, param1=105, param2=62, minRadius=135, maxRadius=195
    )
    
    total_coins = 0
    result_image = original_img.copy()
    if detected_circles is not None:
        total_coins = len(detected_circles[0, :])
        detected_circles = np.uint16(np.around(detected_circles))
        for coin in detected_circles[0, :]:
            cv2.circle(result_image, (coin[0], coin[1]), coin[2], (0, 255, 0), 2)
            cv2.circle(result_image, (coin[0], coin[1]), 2, (0, 0, 255), 3)
    
    return result_image, total_coins

def extract_coins(original_img):
    grayscale = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    clahe_transform = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast_enhanced = clahe_transform.apply(grayscale)
    
    blurred_gray = cv2.GaussianBlur(contrast_enhanced, (5, 5), 0)
    
    _, binary_mask = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    morphology_kernel = np.ones((3, 3), np.uint8)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, morphology_kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    segmented_coins = []
    colorized_output = np.zeros_like(original_img)
    
    for index, contour in enumerate(contours):
        mask_layer = np.zeros_like(grayscale)
        cv2.drawContours(mask_layer, [contour], -1, (255), thickness=cv2.FILLED)
        rand_color = np.random.randint(0, 255, (3,), dtype=int).tolist()
        colorized_output[mask_layer == 255] = rand_color
        x, y, width, height = cv2.boundingRect(contour)
        cropped_coin = original_img[y:y+height, x:x+width]
        segmented_coins.append(cropped_coin)
    
    return colorized_output, segmented_coins

def store_processed_images(img_path, output_directory="./coin_detection_outputs"):
    os.makedirs(output_directory, exist_ok=True)
    
    img, gray, enhanced_gray, blurred = process_input_image(img_path)
    circles_img, detected_coins = locate_coins_hough(blurred, img)
    segmented_result, extracted_coins = extract_coins(img)
    edge_map = cv2.Canny(blurred, 210, 80)
    
    cv2.imwrite(os.path.join(output_directory, "enhanced_grayscale.png"), enhanced_gray)
    cv2.imwrite(os.path.join(output_directory, "hough_detected_circles.png"), circles_img)
    cv2.imwrite(os.path.join(output_directory, "edges_detected.png"), edge_map)
    cv2.imwrite(os.path.join(output_directory, "segmented_result.png"), segmented_result)
    
    for idx, coin_img in enumerate(extracted_coins):
        cv2.imwrite(os.path.join(output_directory, f"coin_segment_{idx+1}.png"), coin_img)
    
    print(f"Images successfully saved in '{output_directory}'")
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Enhanced Grayscale")
    plt.imshow(enhanced_gray, cmap="gray")
    
    plt.subplot(1, 3, 2)
    plt.title("Hough Circle Detection")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(circles_img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 3)
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.imshow(edge_map, cmap="gray")
    plt.show(block=False)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(segmented_result, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Segmented Coins with Colors")
    plt.show(block=False)
    
    total_coins = len(extracted_coins)
    plt.figure(figsize=(12, 6))
    for idx, coin_img in enumerate(extracted_coins):
        plt.subplot(2, total_coins//2 + 1, idx+1)
        plt.imshow(cv2.cvtColor(coin_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Coin {idx+1}")
    plt.show()

img_path = os.path.join(os.getcwd(), "coin_detection_input", "coins.png")
store_processed_images(img_path)
