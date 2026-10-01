import os
import numpy as np
import cv2

from src.ImageLoader import ImageLoader
from src.Filter import Filter
from src.Transform import Transform
from src.utils import load_image, show_image, concatenate_with_labels


# =============================================================================
# PHẦN 1: BIỂU DIỄN ẢNH MÀU VÀ KÊNH MÀU (COLOR & GRAYSCALE REPRESENTATION)
# =============================================================================
def run_color_representation_demo(image_path: str):
    print("\n" + "=" * 60)
    print(f"▶ CHẠY DEMO PHẦN 1: BIỂU DIỄN ẢNH MÀU VÀ KÊNH MÀU: {image_path}")
    print("=" * 60)

    loader = ImageLoader()
    img_list = loader.load_image(image_path)
    if not img_list:
        print(f"Không thể đọc ảnh từ: {image_path}")
        return
    img_obj = img_list[0]

    # 1. Chuyển đổi màu sang xám
    print("[1.1] Chuyển đổi RGB sang Grayscale...")
    gray_img = loader.rgb_to_gray(img_obj)
    if gray_img:
        show_image(gray_img.data)

    # 2. Tách các kênh màu (R, G, B)
    print("[1.2] Tách các kênh màu R, G, B...")
    r_img, g_img, b_img = loader.split_channels(img_obj)
    if r_img:
        print("- Kênh Đỏ (Red Channel):")
        show_image(r_img.data)
        print("- Kênh Xanh lá (Green Channel):")
        show_image(g_img.data)
        print("- Kênh Xanh dương (Blue Channel):")
        show_image(b_img.data)

    # 3. Tái tạo hoặc hoán đổi kênh màu (ví dụ: BGR)
    print("[1.3] Hoán đổi kênh màu (B-G-R) để tạo ảnh màu mới...")
    swapped_img = loader.merge_channels(b_img, g_img, r_img, output_name="swapped_bgr.jpg")
    if swapped_img:
        show_image(swapped_img.data)


# =============================================================================
# PHẦN 2: LỌC ẢNH (FILTER DEMO & SO SÁNH ẢNH GỐC VÀ KẾT QUẢ)
# =============================================================================
def run_filter_demo(image_path: str, base_output_dir: str = "test_sample"):
    print("\n" + "=" * 60)
    print(f"▶ CHẠY DEMO PHẦN 2: LỌC ẢNH VÀ SO SÁNH VỚI ẢNH GỐC: {image_path}")
    print("=" * 60)

    # 1. Tạo subfolder tương ứng theo tên file ảnh
    image_filename = os.path.basename(image_path)
    image_stem = os.path.splitext(image_filename)[0]
    subfolder_path = os.path.join(base_output_dir, image_stem, "filters")
    os.makedirs(subfolder_path, exist_ok=True)

    # 2. Đọc ảnh và chuyển sang ảnh xám
    img_rgb = load_image(image_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    filter_tool = Filter()

    # Lưu ảnh gốc vào subfolder
    cv2.imwrite(
        os.path.join(subfolder_path, "00_original_color.jpg"),
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(subfolder_path, "00_original_gray.jpg"), img_gray)

    # 3. Danh sách các bộ lọc cần thực hiện
    filters_to_run = [
        (
            "01_mean_3x3",
            "Mean Filter (3x3)",
            img_rgb,
            lambda: filter_tool.mean_filter(img_rgb, kernel_size=3),
        ),
        (
            "02_gaussian_5x5",
            "Gaussian Filter (5x5, sigma=1.0)",
            img_rgb,
            lambda: filter_tool.gaussian_filter(img_rgb, kernel_size=5, sigma=1.0),
        ),
        (
            "03_sobel_x",
            "Sobel X (Horizontal Gradient)",
            img_gray,
            lambda: filter_tool.sobel_filter(img_gray, axis="x", kernel_size=3),
        ),
        (
            "04_sobel_y",
            "Sobel Y (Vertical Gradient)",
            img_gray,
            lambda: filter_tool.sobel_filter(img_gray, axis="y", kernel_size=3),
        ),
        (
            "05_sobel_combined",
            "Sobel Magnitude",
            img_gray,
            lambda: filter_tool.sobel_filter(img_gray, axis="both", kernel_size=3),
        ),
        (
            "06_laplacian_4",
            "Laplacian (4-Connectivity)",
            img_gray,
            lambda: filter_tool.laplacian_filter(img_gray, kernel_size=3, connectivity=4),
        ),
        (
            "07_laplacian_8",
            "Laplacian (8-Connectivity)",
            img_gray,
            lambda: filter_tool.laplacian_filter(img_gray, kernel_size=3, connectivity=8),
        ),
    ]

    for save_name, label, orig_input, custom_func in filters_to_run:
        print(f"Đang thực hiện lọc và so sánh: {label}...")
        custom_res = custom_func()

        # Ghép ảnh gốc bên trái và ảnh kết quả bên phải
        comparison_img = concatenate_with_labels(
            custom_img=orig_input,
            cv2_img=custom_res,
            left_label="Anh goc (Original)",
            right_label=f"Ket qua: {label}",
        )

        save_file_path = os.path.join(subfolder_path, f"{save_name}_comparison.jpg")
        cv2.imwrite(save_file_path, comparison_img)

    print(f"\n✔ Hoàn tất demo Filter! Ảnh so sánh được lưu tại: {subfolder_path}")


# =============================================================================
# PHẦN 3: BIẾN ĐỔI HÌNH HỌC (GEOMETRIC TRANSFORMATIONS DEMO & SO SÁNH)
# =============================================================================
def run_transform_demo(image_path: str, base_output_dir: str = "test_sample"):
    print("\n" + "=" * 60)
    print(f"▶ CHẠY DEMO PHẦN 3: BIẾN ĐỔI HÌNH HỌC VÀ SO SÁNH VỚI ẢNH GỐC: {image_path}")
    print("=" * 60)

    image_filename = os.path.basename(image_path)
    image_stem = os.path.splitext(image_filename)[0]
    subfolder_path = os.path.join(base_output_dir, image_stem, "transforms")
    os.makedirs(subfolder_path, exist_ok=True)

    img = load_image(image_path)
    rows, cols = img.shape[:2]
    transform = Transform()

    # Các cặp điểm cho Affine & Projective
    pts_source_aff = np.array([[0, 0], [cols - 1, 0], [0, rows - 1]], dtype=np.float32)
    pts_dest_aff = np.array([[0, 0], [cols - 50, 0], [50, rows - 1]], dtype=np.float32)

    pts_source_proj = np.array([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]], dtype=np.float32)
    pts_dest_proj = np.array([[0, 0], [cols - 1, 0], [50, rows - 1], [cols - 50, rows - 1]], dtype=np.float32)

    transforms_to_run = [
        ("01_translation", "Translation (tx=20, ty=30)", lambda: transform.translation(img, tx=20, ty=30)),
        ("02_rotation", "Rotation (45 deg)", lambda: transform.rotation_v2(img, angle=45)),
        ("03_scaling", "Scaling (0.75x)", lambda: transform.scaling(img, scale_x=0.75, scale_y=0.75)),
        ("04_affine", "Affine Transformation", lambda: transform.affine(img, pts_source_aff, pts_dest_aff)),
        ("05_projective", "Projective Transformation", lambda: transform.projective(img, pts_source_proj, pts_dest_proj)),
    ]

    for save_name, label, trans_func in transforms_to_run:
        print(f"Đang thực hiện biến đổi và so sánh: {label}...")
        res_img = trans_func()

        # Ghép ảnh gốc bên trái và ảnh biến đổi bên phải
        comparison_img = concatenate_with_labels(
            custom_img=img,
            cv2_img=res_img,
            left_label="Anh goc (Original)",
            right_label=f"Ket qua: {label}",
        )

        save_file_path = os.path.join(subfolder_path, f"{save_name}_comparison.jpg")
        cv2.imwrite(save_file_path, comparison_img)

    print(f"\n✔ Hoàn tất demo Transform! Ảnh so sánh được lưu tại: {subfolder_path}")


# =============================================================================
# HÀM ĐIỀU PHỐI CHÍNH (MAIN PIPELINE)
# =============================================================================
def main():
    input_images = [
        "data/sample.jpg",
        "data/pixel.png",
        "test_sample/sample_one/sample_one.jpg",
    ]

    valid_images = [p for p in input_images if os.path.exists(p)]

    if not valid_images:
        print("Lỗi: Không tìm thấy ảnh hợp lệ nào trong đường dẫn kiểm thử!")
        return

    RUN_PART_1 = False
    RUN_PART_2 = True
    RUN_PART_3 = True

    for image_path in valid_images:
        print(f"\n>>> Đang xử lý file: {image_path}")
        if RUN_PART_1:
            run_color_representation_demo(image_path)
        if RUN_PART_2:
            run_filter_demo(image_path)
        if RUN_PART_3:
            run_transform_demo(image_path)


if __name__ == "__main__":
    main()