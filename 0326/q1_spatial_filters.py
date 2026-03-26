import os
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 路径配置
# =========================
image_path = r"D:\class1\1.jpg"
output_dir = r"D:\class1\q1"
os.makedirs(output_dir, exist_ok=True)


# =========================
# 2. 图像读取
# =========================
img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"无法读取图像，请检查路径是否正确：{image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# =========================
# 3. 常见空间滤波
# =========================
# Box Filter / 均值滤波
box_filtered = cv2.blur(img_rgb, (5, 5))

# Gaussian Filter / 高斯滤波
gaussian_filtered = cv2.GaussianBlur(img_rgb, (5, 5), 1.2)

# Median Filter / 中值滤波
median_filtered = cv2.medianBlur(img_rgb, 5)

# Sobel / 边缘检测（对灰度图）
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

sobel_x_show = cv2.convertScaleAbs(sobel_x)
sobel_y_show = cv2.convertScaleAbs(sobel_y)
sobel_mag_show = cv2.convertScaleAbs(sobel_magnitude)


# =========================
# 4. 简单评价指标
# =========================
def calc_metrics(reference: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    ref = reference.astype(np.float64)
    tar = target.astype(np.float64)
    mse = np.mean((ref - tar) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)
    return mse, psnr


mse_box, psnr_box = calc_metrics(img_rgb, box_filtered)
mse_gauss, psnr_gauss = calc_metrics(img_rgb, gaussian_filtered)
mse_median, psnr_median = calc_metrics(img_rgb, median_filtered)


print("=== 空间滤波结果评价（相对原图）===")
print(f"Box Filter     -> MSE: {mse_box:.2f}, PSNR: {psnr_box:.2f} dB")
print(f"Gaussian Filter-> MSE: {mse_gauss:.2f}, PSNR: {psnr_gauss:.2f} dB")
print(f"Median Filter  -> MSE: {mse_median:.2f}, PSNR: {psnr_median:.2f} dB")
print("Sobel 主要用于边缘检测，不适合用与平滑滤波相同的方式比较。")


# =========================
# 5. 保存单张图像
# =========================
cv2.imwrite(os.path.join(output_dir, "original.png"), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "box_5x5.png"), cv2.cvtColor(box_filtered, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "gaussian_5x5_sigma1.2.png"), cv2.cvtColor(gaussian_filtered, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "median_5.png"), cv2.cvtColor(median_filtered, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "sobel_x.png"), sobel_x_show)
cv2.imwrite(os.path.join(output_dir, "sobel_y.png"), sobel_y_show)
cv2.imwrite(os.path.join(output_dir, "sobel_magnitude.png"), sobel_mag_show)


# =========================
# 6. 绘制对比图
# =========================
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(box_filtered)
plt.title("Box Filter")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gaussian_filtered)
plt.title("Gaussian Filter")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(median_filtered)
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(sobel_x_show, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(sobel_mag_show, cmap="gray")
plt.title("Sobel Magnitude")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=300, bbox_inches="tight")
plt.show()

print(f"\n结果已保存到文件夹: {output_dir}")