import os
import imageio.v3 as iio
import numpy as np
import mimetypes

class Image:
    def __init__(self, image_path="", data=None):
        self.path = image_path
        self.data = data

    # property tự động lấy size từ ma trận
    @property
    def height(self):
        return self.data.shape[0] if self.data is not None else 0

    @property
    def width(self):
        return self.data.shape[1] if self.data is not None else 0

class ImageLoader:


    def _process_single_file(self, file_path):
        """Internal helper to read a single image file"""
        try:
            img_data = iio.imread(file_path)
            
            # Đảm bảo dữ liệu luôn ở chuẩn 8-bit (0-255)
            if img_data.dtype != np.uint8:
                raise ValueError("Ảnh đầu vào phải có kiểu uint8.")
                
            if img_data.ndim != 3 or img_data.shape[2] != 3:
                raise ValueError("Ảnh đầu vào phải là ảnh RGB.")
            
            return Image(file_path, img_data)
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
            return None

    def load_image(self, file_path):
        if not os.path.isfile(file_path):
            return None

        file_type, _ = mimetypes.guess_type(file_path)

        if not file_type or not file_type.startswith("image/"):
            return None

        return self._process_single_file(file_path)

    def load_images(self, folder_path):
        images = []

        if not os.path.isdir(folder_path):
            return images

        for name in sorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, name)

            if not os.path.isfile(file_path):
                continue

            image = self.load_image(file_path)

            if image is not None:
                images.append(image)

        return images

    def rgb_to_gray(self, image):
        """Convert RGB image to grayscale using luminance formula."""
        rgb = image.astype(np.float32)
        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def gray_to_rgb(self, image):
        """Convert grayscale image to RGB by stacking the single channel."""
        if image.ndim != 2:
            raise ValueError("Input phải là ảnh grayscale.")

        return np.stack([image, image, image], axis=2)

    
    def split_channels(self, image):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input phải là ảnh RGB.")

        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        return r, g, b

    def merge_channels(self, r, g, b):
        if r.shape != g.shape or r.shape != b.shape:
            raise ValueError("Các kênh phải có cùng kích thước.")

        return np.stack([r, g, b], axis=2)

    
