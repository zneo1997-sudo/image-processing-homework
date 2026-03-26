import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 路径设置
# =========================
image_path = r"D:\class1\1.jpg"
output_dir = r"D:\class1\q3_outputs"
os.makedirs(output_dir, exist_ok=True)


# =========================
# 2. 兼容 Windows/中文路径读取
# =========================
if not os.path.exists(image_path):
    raise FileNotFoundError("找不到图像文件: {}".format(image_path))

img_data = np.fromfile(image_path, dtype=np.uint8)
img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise ValueError("图像读取失败，请检查文件格式")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# =========================
# 3. 傅里叶变换与频谱图
# =========================
def fft_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    spectrum = np.log(1 + magnitude)
    return f, fshift, spectrum


# =========================
# 4. 构造频域滤波器
# =========================
def ideal_lowpass_filter(shape, radius):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            if d <= radius:
                mask[i, j] = 1.0
    return mask


def ideal_highpass_filter(shape, radius):
    return 1.0 - ideal_lowpass_filter(shape, radius)


def gaussian_lowpass_filter(shape, sigma):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    for i in range(rows):
        for j in range(cols):
            d2 = (i - crow) ** 2 + (j - ccol) ** 2
            mask[i, j] = np.exp(-d2 / (2 * sigma * sigma))
    return mask


def gaussian_highpass_filter(shape, sigma):
    return 1.0 - gaussian_lowpass_filter(shape, sigma)


# =========================
# 5. 频域滤波 + 反变换
# =========================
def apply_frequency_filter(image, mask):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    filtered_shift = fshift * mask
    ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(ishift)
    img_back = np.abs(img_back)

    filtered_spectrum = np.log(1 + np.abs(filtered_shift))
    return img_back, filtered_spectrum


# =========================
# 6. 几何变换
# =========================
def translate_image(image, tx, ty):
    rows, cols = image.shape
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows))


def rotate_image(image, angle):
    rows, cols = image.shape
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (cols, rows))


def scale_image(image, scale):
    rows, cols = image.shape
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 为了方便比较，再缩放/裁剪回原尺寸
    result = np.zeros_like(image)
    sh, sw = scaled.shape

    if scale >= 1.0:
        start_y = (sh - rows) // 2
        start_x = (sw - cols) // 2
        result = scaled[start_y:start_y+rows, start_x:start_x+cols]
    else:
        start_y = (rows - sh) // 2
        start_x = (cols - sw) // 2
        result[start_y:start_y+sh, start_x:start_x+sw] = scaled

    return result


# =========================
# 7. 原图频谱
# =========================
_, _, original_spectrum = fft_spectrum(img_gray)


# =========================
# 8. 构造滤波器
# =========================
shape = img_gray.shape
ideal_lp = ideal_lowpass_filter(shape, radius=30)
ideal_hp = ideal_highpass_filter(shape, radius=30)
gaussian_lp = gaussian_lowpass_filter(shape, sigma=30)
gaussian_hp = gaussian_highpass_filter(shape, sigma=30)


# =========================
# 9. 频域滤波
# =========================
img_ideal_lp, spec_ideal_lp = apply_frequency_filter(img_gray, ideal_lp)
img_ideal_hp, spec_ideal_hp = apply_frequency_filter(img_gray, ideal_hp)
img_gaussian_lp, spec_gaussian_lp = apply_frequency_filter(img_gray, gaussian_lp)
img_gaussian_hp, spec_gaussian_hp = apply_frequency_filter(img_gray, gaussian_hp)


# =========================
# 10. 几何变换及频谱比较
# =========================
img_translated = translate_image(img_gray, 50, 30)
img_rotated = rotate_image(img_gray, 30)
img_scaled = scale_image(img_gray, 0.7)

_, _, spectrum_translated = fft_spectrum(img_translated)
_, _, spectrum_rotated = fft_spectrum(img_rotated)
_, _, spectrum_scaled = fft_spectrum(img_scaled)


# =========================
# 11. 保存图像
# =========================
def save_gray(path, image):
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(path, image)


save_gray(os.path.join(output_dir, "original_gray.png"), img_gray)
save_gray(os.path.join(output_dir, "ideal_lowpass_result.png"), img_ideal_lp)
save_gray(os.path.join(output_dir, "ideal_highpass_result.png"), img_ideal_hp)
save_gray(os.path.join(output_dir, "gaussian_lowpass_result.png"), img_gaussian_lp)
save_gray(os.path.join(output_dir, "gaussian_highpass_result.png"), img_gaussian_hp)

save_gray(os.path.join(output_dir, "translated.png"), img_translated)
save_gray(os.path.join(output_dir, "rotated.png"), img_rotated)
save_gray(os.path.join(output_dir, "scaled.png"), img_scaled)


# =========================
# 12. 绘图：频域滤波结果
# =========================
plt.figure(figsize=(16, 10))

plt.subplot(3, 4, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(original_spectrum, cmap="gray")
plt.title("Original Spectrum")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(ideal_lp, cmap="gray")
plt.title("Ideal Lowpass Mask")
plt.axis("off")

plt.subplot(3, 4, 4)
plt.imshow(img_ideal_lp, cmap="gray")
plt.title("Ideal Lowpass Result")
plt.axis("off")

plt.subplot(3, 4, 5)
plt.imshow(ideal_hp, cmap="gray")
plt.title("Ideal Highpass Mask")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(img_ideal_hp, cmap="gray")
plt.title("Ideal Highpass Result")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(gaussian_lp, cmap="gray")
plt.title("Gaussian Lowpass Mask")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(img_gaussian_lp, cmap="gray")
plt.title("Gaussian Lowpass Result")
plt.axis("off")

plt.subplot(3, 4, 9)
plt.imshow(gaussian_hp, cmap="gray")
plt.title("Gaussian Highpass Mask")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.imshow(img_gaussian_hp, cmap="gray")
plt.title("Gaussian Highpass Result")
plt.axis("off")

plt.subplot(3, 4, 11)
plt.imshow(spec_gaussian_lp, cmap="gray")
plt.title("Filtered Spectrum (GLP)")
plt.axis("off")

plt.subplot(3, 4, 12)
plt.imshow(spec_gaussian_hp, cmap="gray")
plt.title("Filtered Spectrum (GHP)")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "frequency_filter_results.png"), dpi=300, bbox_inches="tight")
plt.show()


# =========================
# 13. 绘图：平移/旋转/缩放频谱比较
# =========================
plt.figure(figsize=(14, 8))

plt.subplot(2, 4, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(img_translated, cmap="gray")
plt.title("Translated")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(img_rotated, cmap="gray")
plt.title("Rotated")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(img_scaled, cmap="gray")
plt.title("Scaled")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(original_spectrum, cmap="gray")
plt.title("Original Spectrum")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(spectrum_translated, cmap="gray")
plt.title("Translated Spectrum")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(spectrum_rotated, cmap="gray")
plt.title("Rotated Spectrum")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(spectrum_scaled, cmap="gray")
plt.title("Scaled Spectrum")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "spectrum_transform_compare.png"), dpi=300, bbox_inches="tight")
plt.show()

print("结果已保存到文件夹: {}".format(output_dir))