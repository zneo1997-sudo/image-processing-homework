
import io
import math
from typing import Tuple

import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="模式识别与图像处理课程作业", layout="wide")


# -----------------------------
# 基础工具
# -----------------------------
def read_uploaded_image(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("无法读取上传的图像，请确认文件格式正确。")
    return img_bgr


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img)
    if img.dtype == np.uint8:
        return img
    img = np.abs(img)
    maxv = float(img.max()) if img.size else 0.0
    if maxv == 0:
        return np.zeros_like(img, dtype=np.uint8)
    img = 255.0 * img / maxv
    return np.clip(img, 0, 255).astype(np.uint8)


def calc_metrics(reference: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    ref = reference.astype(np.float64)
    tar = target.astype(np.float64)
    mse = float(np.mean((ref - tar) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10((255 ** 2) / mse))
    return mse, psnr


def fft_spectrum(image_gray: np.ndarray):
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    spectrum = np.log1p(magnitude)
    return fshift, spectrum


def ideal_lowpass_filter(shape, radius):
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows // 2, cols // 2
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    return (dist <= radius).astype(np.float32)


def ideal_highpass_filter(shape, radius):
    return 1.0 - ideal_lowpass_filter(shape, radius)


def gaussian_lowpass_filter(shape, sigma):
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    cy, cx = rows // 2, cols // 2
    dist2 = (y - cy) ** 2 + (x - cx) ** 2
    return np.exp(-dist2 / (2 * sigma * sigma)).astype(np.float32)


def gaussian_highpass_filter(shape, sigma):
    return 1.0 - gaussian_lowpass_filter(shape, sigma)


def apply_frequency_filter(image_gray: np.ndarray, mask: np.ndarray):
    fshift, _ = fft_spectrum(image_gray)
    filtered_shift = fshift * mask
    inv = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    restored = np.abs(inv)
    filtered_spectrum = np.log1p(np.abs(filtered_shift))
    return restored, filtered_spectrum


def translate_image(image_gray: np.ndarray, tx: int, ty: int):
    rows, cols = image_gray.shape
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image_gray, m, (cols, rows))


def rotate_image(image_gray: np.ndarray, angle: float):
    rows, cols = image_gray.shape
    center = (cols // 2, rows // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image_gray, m, (cols, rows))


def scale_image(image_gray: np.ndarray, scale: float):
    rows, cols = image_gray.shape
    scaled = cv2.resize(image_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    result = np.zeros_like(image_gray)
    sh, sw = scaled.shape

    if scale >= 1.0:
        start_y = max((sh - rows) // 2, 0)
        start_x = max((sw - cols) // 2, 0)
        return scaled[start_y:start_y + rows, start_x:start_x + cols]
    else:
        start_y = max((rows - sh) // 2, 0)
        start_x = max((cols - sw) // 2, 0)
        end_y = min(start_y + sh, rows)
        end_x = min(start_x + sw, cols)
        result[start_y:end_y, start_x:end_x] = scaled[:end_y - start_y, :end_x - start_x]
        return result


# -----------------------------
# 页面标题
# -----------------------------
st.title("模式识别与图像处理课程作业 Web 应用")
st.caption("支持：空间滤波、局部区域梯度分析、频率域滤波与频谱变化比较")

uploaded = st.file_uploader("上传一张图片", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is None:
    st.info("请先上传图片。建议上传清晰一些的图像，便于观察滤波与频谱变化。")
    st.stop()

img_bgr = read_uploaded_image(uploaded)
img_rgb = bgr_to_rgb(img_bgr)
img_gray = to_gray(img_bgr)

h, w = img_gray.shape

with st.expander("查看原图", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption=f"原图 RGB | 尺寸: {w} x {h}", use_container_width=True)
    with col2:
        st.image(img_gray, caption="原图灰度图", clamp=True, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["第一题：空间滤波", "第二题：局部区域梯度", "第三题：频率域滤波"])

# -----------------------------
# 第一题：空间滤波
# -----------------------------
with tab1:
    st.subheader("常见空间图像滤波器比较")
    c1, c2, c3 = st.columns(3)

    with c1:
        box_ksize = st.slider("Box 核大小", 3, 15, 5, step=2)
        median_ksize = st.slider("Median 核大小", 3, 15, 5, step=2)

    with c2:
        gauss_ksize = st.slider("Gaussian 核大小", 3, 15, 5, step=2)
        gauss_sigma = st.slider("Gaussian sigma", 0.1, 10.0, 1.2, step=0.1)

    with c3:
        sobel_ksize = st.selectbox("Sobel 核大小", [1, 3, 5, 7], index=1)

    box_filtered = cv2.blur(img_rgb, (box_ksize, box_ksize))
    gauss_filtered = cv2.GaussianBlur(img_rgb, (gauss_ksize, gauss_ksize), gauss_sigma)
    median_filtered = cv2.medianBlur(img_rgb, median_ksize)

    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    mse_box, psnr_box = calc_metrics(img_rgb, box_filtered)
    mse_gauss, psnr_gauss = calc_metrics(img_rgb, gauss_filtered)
    mse_median, psnr_median = calc_metrics(img_rgb, median_filtered)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Box PSNR", f"{psnr_box:.2f} dB", f"MSE={mse_box:.2f}")
    mc2.metric("Gaussian PSNR", f"{psnr_gauss:.2f} dB", f"MSE={mse_gauss:.2f}")
    mc3.metric("Median PSNR", f"{psnr_median:.2f} dB", f"MSE={mse_median:.2f}")

    fig1, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(box_filtered)
    axes[0, 1].set_title("Box")
    axes[0, 2].imshow(gauss_filtered)
    axes[0, 2].set_title("Gaussian")
    axes[1, 0].imshow(median_filtered)
    axes[1, 0].set_title("Median")
    axes[1, 1].imshow(normalize_to_uint8(sobel_x), cmap="gray")
    axes[1, 1].set_title("Sobel X")
    axes[1, 2].imshow(normalize_to_uint8(sobel_mag), cmap="gray")
    axes[1, 2].set_title("Sobel Magnitude")
    for ax in axes.ravel():
        ax.axis("off")
    fig1.tight_layout()
    st.pyplot(fig1)

    st.markdown(
        """
        **比较提示**
        - Box：平滑明显，但容易模糊边缘。
        - Gaussian：平滑更自然，通常比 Box 更柔和。
        - Median：对椒盐噪声更有效，保边能力通常更好。
        - Sobel：主要用于边缘与梯度检测，不属于纯平滑滤波。
        """
    )

# -----------------------------
# 第二题：梯度分析
# -----------------------------
with tab2:
    st.subheader("局部区域梯度方向演示")
    st.write("通过调整 ROI 和步长，观察局部区域的 Gx、Gy、梯度幅值与方向场。")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        x = st.slider("ROI 左上角 x", 0, max(0, w - 2), min(20, max(0, w - 2)))
        y = st.slider("ROI 左上角 y", 0, max(0, h - 2), min(20, max(0, h - 2)))
        roi_w = st.slider("ROI 宽度", 20, max(20, w - x), min(180, max(20, w - x)))
        roi_h = st.slider("ROI 高度", 20, max(20, h - y), min(180, max(20, h - y)))

    with col_b:
        grad_ksize = st.selectbox("梯度 Sobel 核大小", [1, 3, 5, 7], index=1, key="grad_sobel")
        step = st.slider("箭头采样步长", 4, 30, 12)
        mag_threshold = st.slider("方向显示最小幅值阈值", 0, 255, 20)

    roi_w = min(roi_w, w - x)
    roi_h = min(roi_h, h - y)

    roi_gray = img_gray[y:y + roi_h, x:x + roi_w]
    roi_rgb = img_rgb[y:y + roi_h, x:x + roi_w].copy()

    gx = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=grad_ksize)
    gy = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=grad_ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle_rad = np.arctan2(gy, gx)
    angle_deg = np.degrees(angle_rad)

    cos_mean = np.sum(np.cos(angle_rad) * magnitude) / (np.sum(magnitude) + 1e-8)
    sin_mean = np.sum(np.sin(angle_rad) * magnitude) / (np.sum(magnitude) + 1e-8)
    main_angle_deg = float(np.degrees(np.arctan2(sin_mean, cos_mean)))

    overlay = img_rgb.copy()
    cv2.rectangle(overlay, (x, y), (x + roi_w, y + roi_h), (255, 0, 0), 2)

    yy, xx = np.mgrid[0:roi_h:step, 0:roi_w:step]
    u = gx[::step, ::step]
    v = gy[::step, ::step]
    m = magnitude[::step, ::step]
    mask = m >= mag_threshold
    norm = np.sqrt(u ** 2 + v ** 2) + 1e-8
    u_norm = u / norm
    v_norm = v / norm

    m1, m2, m3 = st.columns(3)
    m1.metric("主梯度方向", f"{main_angle_deg:.2f}°")
    m2.metric("梯度均值", f"{float(np.mean(magnitude)):.2f}")
    m3.metric("梯度最大值", f"{float(np.max(magnitude)):.2f}")

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    axes2[0, 0].imshow(overlay)
    axes2[0, 0].set_title("Original with ROI")
    axes2[0, 1].imshow(roi_rgb)
    axes2[0, 1].set_title("Local ROI")
    axes2[0, 2].imshow(normalize_to_uint8(gx), cmap="gray")
    axes2[0, 2].set_title("Gx")

    axes2[1, 0].imshow(normalize_to_uint8(gy), cmap="gray")
    axes2[1, 0].set_title("Gy")
    axes2[1, 1].imshow(normalize_to_uint8(magnitude), cmap="gray")
    axes2[1, 1].set_title("Magnitude")
    axes2[1, 2].imshow(roi_gray, cmap="gray")
    axes2[1, 2].quiver(
        xx[mask],
        yy[mask],
        u_norm[mask],
        -v_norm[mask],
        angles="xy",
        scale_units="xy",
        scale=0.35,
    )
    axes2[1, 2].set_title(f"Direction Field | Main={main_angle_deg:.2f}°")

    for ax in axes2.ravel():
        ax.axis("off")
    fig2.tight_layout()
    st.pyplot(fig2)

    st.write("说明：梯度方向表示灰度增长最快的方向，梯度幅值较大的位置通常对应明显边缘。")

# -----------------------------
# 第三题：频率域滤波
# -----------------------------
with tab3:
    st.subheader("频率域图像滤波与频谱变化比较")

    left, right = st.columns([1, 1])

    with left:
        filter_type = st.selectbox(
            "选择频域滤波器",
            ["Ideal Lowpass", "Ideal Highpass", "Gaussian Lowpass", "Gaussian Highpass"],
        )
        radius = st.slider("理想滤波截止半径", 5, 200, 30)
        sigma = st.slider("高斯滤波 sigma", 5, 200, 30)
    with right:
        tx = st.slider("平移 tx", -200, 200, 50)
        ty = st.slider("平移 ty", -200, 200, 30)
        angle = st.slider("旋转角度", -180, 180, 30)
        scale = st.slider("缩放倍数", 0.2, 2.0, 0.7, step=0.1)

    original_shift, original_spectrum = fft_spectrum(img_gray)

    if filter_type == "Ideal Lowpass":
        mask = ideal_lowpass_filter(img_gray.shape, radius)
    elif filter_type == "Ideal Highpass":
        mask = ideal_highpass_filter(img_gray.shape, radius)
    elif filter_type == "Gaussian Lowpass":
        mask = gaussian_lowpass_filter(img_gray.shape, sigma)
    else:
        mask = gaussian_highpass_filter(img_gray.shape, sigma)

    filtered_img, filtered_spectrum = apply_frequency_filter(img_gray, mask)

    translated = translate_image(img_gray, tx, ty)
    rotated = rotate_image(img_gray, angle)
    scaled = scale_image(img_gray, scale)

    _, translated_spectrum = fft_spectrum(translated)
    _, rotated_spectrum = fft_spectrum(rotated)
    _, scaled_spectrum = fft_spectrum(scaled)

    fig3, axes3 = plt.subplots(2, 3, figsize=(12, 8))
    axes3[0, 0].imshow(img_gray, cmap="gray")
    axes3[0, 0].set_title("Original")
    axes3[0, 1].imshow(original_spectrum, cmap="gray")
    axes3[0, 1].set_title("Original Spectrum")
    axes3[0, 2].imshow(mask, cmap="gray")
    axes3[0, 2].set_title("Frequency Mask")

    axes3[1, 0].imshow(normalize_to_uint8(filtered_img), cmap="gray")
    axes3[1, 0].set_title("Filtered Result")
    axes3[1, 1].imshow(filtered_spectrum, cmap="gray")
    axes3[1, 1].set_title("Filtered Spectrum")
    axes3[1, 2].axis("off")

    for ax in axes3.ravel():
        if ax.has_data():
            ax.axis("off")
    fig3.tight_layout()
    st.pyplot(fig3)

    fig4, axes4 = plt.subplots(2, 4, figsize=(14, 8))
    axes4[0, 0].imshow(img_gray, cmap="gray")
    axes4[0, 0].set_title("Original")
    axes4[0, 1].imshow(translated, cmap="gray")
    axes4[0, 1].set_title("Translated")
    axes4[0, 2].imshow(rotated, cmap="gray")
    axes4[0, 2].set_title("Rotated")
    axes4[0, 3].imshow(scaled, cmap="gray")
    axes4[0, 3].set_title("Scaled")

    axes4[1, 0].imshow(original_spectrum, cmap="gray")
    axes4[1, 0].set_title("Original Spectrum")
    axes4[1, 1].imshow(translated_spectrum, cmap="gray")
    axes4[1, 1].set_title("Translated Spectrum")
    axes4[1, 2].imshow(rotated_spectrum, cmap="gray")
    axes4[1, 2].set_title("Rotated Spectrum")
    axes4[1, 3].imshow(scaled_spectrum, cmap="gray")
    axes4[1, 3].set_title("Scaled Spectrum")

    for ax in axes4.ravel():
        ax.axis("off")
    fig4.tight_layout()
    st.pyplot(fig4)

    st.markdown(
        """
        **观察结论**
        - 平移：主要改变相位，幅度谱通常基本不变。
        - 旋转：频谱会随图像同步旋转。
        - 缩放：空间域放大对应频域压缩，空间域缩小对应频域扩展。
        - 低通：图像更平滑。
        - 高通：边缘和细节更突出。
        """
    )
