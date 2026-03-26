import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 路径设置
# =========================
image_path = r"D:\class1\1.jpg"
output_dir = r"D:\class1\q2_outputs"
os.makedirs(output_dir, exist_ok=True)


# =========================
# 2. 兼容中文/Windows路径读取
# =========================
if not os.path.exists(image_path):
    raise FileNotFoundError("找不到图像文件: {}".format(image_path))

img_data = np.fromfile(image_path, dtype=np.uint8)
img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

if img_bgr is None:
    raise ValueError("图像读取失败，请检查文件格式是否正确")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# =========================
# 3. 手动设置局部区域 ROI
#    你可以修改这里
# =========================
# 格式: x, y, w, h
x, y, w, h = 150, 120, 180, 180

# 防止越界
H, W = img_gray.shape
x = max(0, min(x, W - 1))
y = max(0, min(y, H - 1))
w = min(w, W - x)
h = min(h, H - y)

roi_gray = img_gray[y:y+h, x:x+w]
roi_rgb = img_rgb[y:y+h, x:x+w].copy()


# =========================
# 4. Sobel 计算梯度
# =========================
gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)

magnitude = np.sqrt(gx**2 + gy**2)
angle_rad = np.arctan2(gy, gx)
angle_deg = np.degrees(angle_rad)

# 归一化显示
gx_show = cv2.convertScaleAbs(gx)
gy_show = cv2.convertScaleAbs(gy)
mag_show = cv2.convertScaleAbs(magnitude)


# =========================
# 5. 计算该区域整体主方向
# =========================
# 方法：使用梯度幅值作为权重，对方向向量求平均
cos_mean = np.sum(np.cos(angle_rad) * magnitude) / (np.sum(magnitude) + 1e-8)
sin_mean = np.sum(np.sin(angle_rad) * magnitude) / (np.sum(magnitude) + 1e-8)
main_angle_rad = np.arctan2(sin_mean, cos_mean)
main_angle_deg = np.degrees(main_angle_rad)

print("=== 局部区域梯度分析 ===")
print("ROI = x:{}, y:{}, w:{}, h:{}".format(x, y, w, h))
print("平均主梯度方向(角度): {:.2f}°".format(main_angle_deg))
print("梯度幅值均值: {:.2f}".format(np.mean(magnitude)))
print("梯度幅值最大值: {:.2f}".format(np.max(magnitude)))


# =========================
# 6. 在原图上框出 ROI
# =========================
img_with_box = img_rgb.copy()
cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (255, 0, 0), 2)


# =========================
# 7. 梯度方向箭头图
# =========================
# 为了避免箭头太密，这里做降采样
step = 12
yy, xx = np.mgrid[0:h:step, 0:w:step]

u = gx[::step, ::step]
v = gy[::step, ::step]

# 归一化箭头长度，避免过长
norm = np.sqrt(u**2 + v**2) + 1e-8
u_norm = u / norm
v_norm = v / norm


# =========================
# 8. 保存图像
# =========================
cv2.imwrite(os.path.join(output_dir, "original_with_roi.png"),
            cv2.cvtColor(img_with_box, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "roi.png"),
            cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "gx.png"), gx_show)
cv2.imwrite(os.path.join(output_dir, "gy.png"), gy_show)
cv2.imwrite(os.path.join(output_dir, "magnitude.png"), mag_show)


# =========================
# 9. 绘图展示
# =========================
plt.figure(figsize=(14, 10))

plt.subplot(2, 3, 1)
plt.imshow(img_with_box)
plt.title("Original Image with ROI")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(roi_rgb)
plt.title("Local ROI")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gx_show, cmap="gray")
plt.title("Gradient Gx")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gy_show, cmap="gray")
plt.title("Gradient Gy")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(mag_show, cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(roi_gray, cmap="gray")
plt.quiver(xx, yy, u_norm, -v_norm, angles='xy', scale_units='xy', scale=0.4)
plt.title("Gradient Direction Field\nMain Angle = {:.2f}°".format(main_angle_deg))
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gradient_demo.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\n结果已保存到文件夹: {}".format(output_dir))