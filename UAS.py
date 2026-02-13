import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

IMG1_PATH = "images/1.jpeg"
IMG2_PATH = "images/2.jpeg"

img1 = cv2.imread(IMG1_PATH)
img2 = cv2.imread(IMG2_PATH)

if img1 is None:
    raise FileNotFoundError(f"Gagal membaca {IMG1_PATH}. Pastikan file ada & path benar.")
if img2 is None:
    raise FileNotFoundError(f"Gagal membaca {IMG2_PATH}. Pastikan file ada & path benar.")

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

try:
    sift = cv2.SIFT_create()
except AttributeError:
    raise RuntimeError("SIFT tidak tersedia. Install: pip install opencv-contrib-python")

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print("Keypoints img1:", len(kp1))
print("Keypoints img2:", len(kp2))

if des1 is None or des2 is None:
    raise RuntimeError("Descriptor kosong (des1/des2 None). Coba gambar lebih jelas/bertekstur.")

kp_img1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
kp_img2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite("outputs/keypoints_1.png", kp_img1)
cv2.imwrite("outputs/keypoints_2.png", kp_img2)
print("Saved: outputs/keypoints_1.png, outputs/keypoints_2.png")

bf = cv2.BFMatcher(cv2.NORM_L2)
raw_matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for pair in raw_matches:
    if len(pair) < 2:
        continue
    m, n = pair
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good matches:", len(good_matches))

if len(good_matches) < 4:
    raise RuntimeError("Good matches < 4, homography tidak bisa dihitung.")

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
if H is None or mask is None:
    raise RuntimeError("Homography gagal dihitung (H/mask None).")

mask = mask.ravel().astype(bool)
inlier_matches = [m for m, inl in zip(good_matches, mask) if inl]
print("Inlier matches:", len(inlier_matches))

if len(inlier_matches) < 4:
    raise RuntimeError("Inlier matches < 4, hasil homography tidak stabil.")

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

corners_img1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
corners_img2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

warped_corners_img1 = cv2.perspectiveTransform(corners_img1, H)
all_corners = np.concatenate((warped_corners_img1, corners_img2), axis=0)

x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

panorama_width = int(x_max - x_min)
panorama_height = int(y_max - y_min)

T = np.array([[1, 0, -x_min],
              [0, 1, -y_min],
              [0, 0, 1]], dtype=np.float32)

panorama = cv2.warpPerspective(img1, T @ H, (panorama_width, panorama_height))

x_offset = -x_min
y_offset = -y_min

#area tempel
x1 = max(x_offset, 0)
y1 = max(y_offset, 0)
x2 = min(x_offset + w2, panorama_width)
y2 = min(y_offset + h2, panorama_height)

roi = panorama[y1:y2, x1:x2]
img2_crop = img2[(y1 - y_offset):(y2 - y_offset), (x1 - x_offset):(x2 - x_offset)]

#area ROI hytam dengan img2
mask_black = (roi.sum(axis=2) == 0)
roi[mask_black] = img2_crop[mask_black]
panorama[y1:y2, x1:x2] = roi

# save panorama
cv2.imwrite("outputs/panorama.png", panorama)
print("Saved: outputs/panorama.png")

panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

matches_vis = cv2.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    inlier_matches[:100],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(14, 8))
plt.imshow(matches_vis)
plt.axis("off")
plt.title("SIFT Inlier Matches")
plt.tight_layout()
plt.savefig("outputs/matches.png", dpi=300)
plt.close()
print("Saved: outputs/matches.png")

plt.figure(figsize=(18, 12))

plt.subplot(3, 2, 1)
plt.imshow(img1_rgb)
plt.title("Before: Image 1")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(img2_rgb)
plt.title("Before: Image 2")
plt.axis("off")

plt.subplot(3, 2, (3, 4))
plt.imshow(matches_vis)
plt.title("SIFT Feature Matching (Inliers)")
plt.axis("off")

plt.subplot(3, 2, (5, 6))
plt.imshow(panorama_rgb)
plt.title("After: Panorama")
plt.axis("off")

plt.tight_layout()
plt.show()
