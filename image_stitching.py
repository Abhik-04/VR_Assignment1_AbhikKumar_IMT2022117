import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Update matplotlib font size
plt.rcParams.update({'font.size': 18})

# Create output directory
save_dir = "image_stitching_outputs"
os.makedirs(save_dir, exist_ok=True)

def compute_homography(kpts1, kpts2, matches):
    """Calculates the Homography matrix using matched keypoints."""
    if len(matches) < 4:
        raise ValueError("Insufficient matches detected!")
    src_points = np.float32([kpts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kpts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H_matrix, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H_matrix, mask

def crop_black_borders(img):
    """Removes black regions from the stitched image."""
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def merge_images(left_img, right_img, H_matrix):
    """Combines two images using Homography and Blending."""
    h_left, w_left = left_img.shape[:2]
    h_right, w_right = right_img.shape[:2]

    corners_left = np.float32([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]]).reshape(-1, 1, 2)
    corners_transformed = cv2.perspectiveTransform(corners_left, H_matrix)
    all_corners = np.concatenate((corners_left, corners_transformed), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 1)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 1)

    shift = [-x_min, -y_min]
    transform_matrix = np.array([[1, 0, shift[0]],
                                  [0, 1, shift[1]],
                                  [0, 0, 1]], dtype=np.float32)

    warped_left = cv2.warpPerspective(left_img, transform_matrix.dot(H_matrix), (x_max - x_min, y_max - y_min))
    warped_right = cv2.warpPerspective(right_img, transform_matrix, (x_max - x_min, y_max - y_min))

    mask_l = np.tile(np.linspace(1, 0, w_left, dtype=np.float32), (h_left, 1))
    mask_r = np.tile(np.linspace(0, 1, w_right, dtype=np.float32), (h_right, 1))

    mask_l = cv2.warpPerspective(mask_l, transform_matrix.dot(H_matrix), (x_max - x_min, y_max - y_min))
    mask_r = cv2.warpPerspective(mask_r, transform_matrix, (x_max - x_min, y_max - y_min))

    mask_l = np.repeat(mask_l[:, :, np.newaxis], 3, axis=2)
    mask_r = np.repeat(mask_r[:, :, np.newaxis], 3, axis=2)

    blended_result = (warped_left * mask_l + warped_right * mask_r) / (mask_l + mask_r + 1e-10)
    blended_result = np.clip(blended_result, 0, 255).astype(np.uint8)
    return blended_result

def add_padding(imgs, pad_size=120):
    """Adds spacing between concatenated images."""
    height, width, _ = imgs[0].shape
    pad_block = np.full((height, pad_size, 3), 255, dtype=np.uint8)
    final_output = imgs[0]
    for img in imgs[1:]:
        final_output = np.concatenate((final_output, pad_block, img), axis=1)
    return final_output

# Load input images
img_list = []
for idx in range(1, 4):
    image = cv2.imread(os.path.join(os.getcwd(), "image_stitching_input", f"./img{idx}.png"))
    image = cv2.resize(image, (850, 650))
    img_list.append(image)

feature_extractor = cv2.SIFT_create()
keypoints_list, descriptors_list, kp_images = [], [], []

for index, img in enumerate(img_list):
    kp, des = feature_extractor.detectAndCompute(img, None)
    keypoints_list.append(kp)
    descriptors_list.append(des)
    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    kp_images.append(img_with_kp)
    cv2.imwrite(os.path.join(save_dir, f"keypoints_img{index+1}.png"), img_with_kp)

matcher = cv2.BFMatcher(cv2.NORM_L2)
knn_matches1 = matcher.knnMatch(descriptors_list[0], descriptors_list[1], k=2)
knn_matches2 = matcher.knnMatch(descriptors_list[1], descriptors_list[2], k=2)

valid_matches1 = [m for m, n in knn_matches1 if m.distance < 0.75 * n.distance]
valid_matches2 = [m for m, n in knn_matches2 if m.distance < 0.75 * n.distance]

H1, _ = compute_homography(keypoints_list[0], keypoints_list[1], valid_matches1)
stitched_1 = merge_images(img_list[0], img_list[1], H1)

kp_stitched, des_stitched = feature_extractor.detectAndCompute(stitched_1, None)
knn_matches3 = matcher.knnMatch(des_stitched, descriptors_list[2], k=2)
valid_matches3 = [m for m, n in knn_matches3 if m.distance < 0.75 * n.distance]

H2, _ = compute_homography(kp_stitched, keypoints_list[2], valid_matches3)
stiched_final = merge_images(stitched_1, img_list[2], H2)
stiched_final = crop_black_borders(stiched_final)

cv2.imwrite(os.path.join(save_dir, "stitched_result.png"), stiched_final)
print(f"Images successfully saved in '{save_dir}'")

img_rgb1 = cv2.cvtColor(kp_images[0], cv2.COLOR_BGR2RGB)
img_rgb2 = cv2.cvtColor(kp_images[1], cv2.COLOR_BGR2RGB)
img_rgb3 = cv2.cvtColor(kp_images[2], cv2.COLOR_BGR2RGB)
stitched_rgb = cv2.cvtColor(stiched_final, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 2]})
padded_images = add_padding([img_rgb1, img_rgb2, img_rgb3], pad_size=180)

axs[0].imshow(padded_images)
axs[0].set_title("Original Images (with Keypoints)")
axs[0].axis("off")

axs[1].imshow(stitched_rgb)
axs[1].set_title("Final Stitched Panorama")
axs[1].axis("off")

plt.tight_layout()
plt.show()
