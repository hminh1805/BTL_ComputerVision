from src.ImageLoader import ImageLoader, Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    try:
        loader = ImageLoader()
        images = loader.load_image(image_path)
        if images and images[0].data is not None:
            return images[0].data
    except Exception:
        pass

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def preprocess_image(image, to_float32=False):
    """
    Tiền xử lý dữ liệu ảnh đầu vào: hỗ trợ đường dẫn (str), đối tượng Image (ImageLoader),
    hoặc mảng NumPy (np.ndarray).
    Trả về ảnh xám chuẩn dạng 2D (H, W) hoặc ảnh màu 3D (H, W, 3).
    """
    if isinstance(image, str):
        image = load_image(image)

    raw_img = image.data if isinstance(image, Image) or hasattr(image, "data") else image

    if raw_img is None:
        raise ValueError("Ảnh đầu vào không hợp lệ hoặc rỗng !!!")

    img_arr = np.array(raw_img)

    # Khử kênh alpha nếu có (4 kênh RGBA -> 3 kênh RGB)
    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = cv2.cvtColor(img_arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    elif img_arr.ndim == 3 and img_arr.shape[2] == 1:
        img_arr = img_arr.squeeze(axis=2)

    # Đảm bảo dải giá trị [0, 255] nếu ảnh float [0.0, 1.0]
    if np.issubdtype(img_arr.dtype, np.floating):
        if img_arr.size > 0 and img_arr.max() <= 1.0:
            img_arr = img_arr * 255.0

    if to_float32:
        return img_arr.astype(np.float32)
    return img_arr
    
    
def rgb_to_gray(image: np.ndarray):
    if image is None:
        raise ValueError("Ảnh đầu vào không hợp lệ.")
    gray_channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray_channel, cv2.COLOR_GRAY2RGB)

def show_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()

import cv2
import numpy as np


def concatenate_with_labels(
    custom_img: np.ndarray,
    cv2_img: np.ndarray,
    left_label: str = "Custom Filter",
    right_label: str = "OpenCV (cv2)",
) -> np.ndarray:
  """Ghép 2 ảnh cạnh nhau theo chiều ngang và chèn header hiển thị tên phương pháp."""
  # Đồng bộ số kênh: Nếu ảnh 2D (grayscale) thì chuyển sang BGR 3 kênh để vẽ chữ và ghép
  if custom_img.ndim == 2:
    custom_disp = cv2.cvtColor(custom_img, cv2.COLOR_GRAY2BGR)
  elif custom_img.shape[2] == 3:
    custom_disp = cv2.cvtColor(custom_img, cv2.COLOR_RGB2BGR)
  else:
    custom_disp = custom_img.copy()

  if cv2_img.ndim == 2:
    cv2_disp = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
  elif cv2_img.shape[2] == 3:
    cv2_disp = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
  else:
    cv2_disp = cv2_img.copy()

  # Đồng bộ kích thước chiều cao nếu có chênh lệch
  h1, w1 = custom_disp.shape[:2]
  h2, w2 = cv2_disp.shape[:2]
  if h1 != h2:
    cv2_disp = cv2.resize(cv2_disp, (w1, h1))
    h2, w2 = h1, w1

  # Ghép 2 ảnh nằm ngang
  combined = np.hstack([custom_disp, cv2_disp])

  # Tạo banner header màu đen phía trên để ghi chữ
  header_height = 40
  header = np.zeros((header_height, combined.shape[1], 3), dtype=np.uint8)

  # Ghi chữ nhãn cho Custom bên trái và cv2 bên phải
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(
      header,
      left_label,
      (20, 26),
      font,
      0.7,
      (0, 255, 255),
      2,
      lineType=cv2.LINE_AA,
  )
  cv2.putText(
      header,
      right_label,
      (w1 + 20, 26),
      font,
      0.7,
      (0, 255, 0),
      2,
      lineType=cv2.LINE_AA,
  )

  # Nối header với ảnh kết quả
  final_result = np.vstack([header, combined])
  return final_result