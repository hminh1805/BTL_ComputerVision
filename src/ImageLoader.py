import os
import imageio.v3 as iio
import numpy as np
import mimetypes
import cv2

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
    def __init__(self):
        self.image_list = []

    @staticmethod
    def _to_uint8(image_data):
        image_data = np.asarray(image_data)
        if image_data.dtype == np.uint8:
            return image_data
        if np.issubdtype(image_data.dtype, np.floating):
            if image_data.size and image_data.min() >= 0 and image_data.max() <= 1:
                image_data = image_data * 255
        elif np.issubdtype(image_data.dtype, np.integer):
            image_data = image_data.astype(np.float32) * (255.0 / np.iinfo(image_data.dtype).max)
        return np.clip(image_data, 0, 255).astype(np.uint8)

    def _process_single_file(self, file_path):
        try:
            image_data = self._to_uint8(iio.imread(file_path))
            if image_data.ndim == 3 and image_data.shape[2] == 4:
                image_data = image_data[:, :, :3]
            if image_data.ndim not in (2, 3) or (image_data.ndim == 3 and image_data.shape[2] != 3):
                raise ValueError("Ảnh phải là ảnh xám hoặc ảnh RGB.")
            return Image(file_path, image_data)
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
            return None

    def _process_with_opencv(self, file_path):
        image_data = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image_data is None:
            return None
        if image_data.ndim == 3:
            if image_data.shape[2] == 4:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
            else:
                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return Image(file_path, self._to_uint8(image_data))

    def load_image(self, path, source="custom"):
        source = source.lower()
        if source not in ("custom", "opencv"):
            raise ValueError("source phải là 'custom' hoặc 'opencv'.")
        self.image_list = []

        if os.path.isfile(path):
            file_type, _ = mimetypes.guess_type(path)
            if file_type and file_type.startswith('image/'):
                img_obj = self._process_single_file(path) if source == "custom" else self._process_with_opencv(path)
                if img_obj: self.image_list.append(img_obj)
            else:
                print(f"Bỏ qua file không phải ảnh: {path}")
            return self.image_list

        elif os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                path_file = os.path.join(path, name)
                if not os.path.isfile(path_file):
                    continue
                file_type, _ = mimetypes.guess_type(path_file)
                if file_type and file_type.startswith('image/'):
                    img_obj = self._process_single_file(path_file) if source == "custom" else self._process_with_opencv(path_file)
                    if img_obj:
                        self.image_list.append(img_obj)
            print(f"Đã load thành công {len(self.image_list)} ảnh.")
            return self.image_list
        else:
            print("Đường dẫn không hợp lệ!")
            return []

    def rgb_to_gray(self, img_obj):
        """Convert RGB image to grayscale using luminance formula."""
        try:
            if img_obj is None or img_obj.data is None:
                raise ValueError("Ảnh đầu vào không hợp lệ.")
            if img_obj.data.ndim == 2:
                return Image(f"gray_{os.path.basename(img_obj.path)}", img_obj.data.copy())
            if img_obj.data.ndim != 3 or img_obj.data.shape[2] != 3:
                raise ValueError("Ảnh đầu vào phải là ảnh RGB hoặc ảnh xám.")
            rgb = img_obj.data.astype(np.float32)

            r = rgb[:, :, 0]
            g = rgb[:, :, 1]
            b = rgb[:, :, 2]

            gray = 0.299 * r + 0.587 * g + 0.114 * b

            gray = np.clip(gray, 0, 255).astype(np.uint8)

            return Image(
                f"gray_{os.path.basename(img_obj.path)}",
                gray
            )

        except Exception as e:
            print(f"Error in rgb_to_gray: {e}")
            return None

    def gray_to_rgb(self, img_obj):
        try:
            gray = img_obj.data

            if gray.ndim != 2:
                raise ValueError("Input phải là ảnh grayscale.")

            height, width = gray.shape

            rgb = np.zeros((height, width, 3), dtype=np.uint8)

            rgb[:, :, 0] = gray  # Red
            rgb[:, :, 1] = gray  # Green
            rgb[:, :, 2] = gray  # Blue

            return Image(
                f"rgb_{os.path.basename(img_obj.path)}",
                rgb
            )

        except Exception as e:
            print(f"Error in gray_to_rgb: {e}")
            return None
    
    def split_channels(self, img_obj):
        """Split RGB image into R, G, B channels as separate grayscale images"""
        try:
            if img_obj is None or img_obj.data is None or img_obj.data.ndim != 3 or img_obj.data.shape[2] != 3:
                raise ValueError("Ảnh đầu vào phải là ảnh RGB.")
            r_data = img_obj.data[:, :, 0]
            g_data = img_obj.data[:, :, 1]
            b_data = img_obj.data[:, :, 2]
            
            r_img = Image(f"R_{os.path.basename(img_obj.path)}", r_data)
            g_img = Image(f"G_{os.path.basename(img_obj.path)}", g_data)
            b_img = Image(f"B_{os.path.basename(img_obj.path)}", b_data)
            
            return r_img, g_img, b_img
        except Exception as e:
            return None, None, None

    def merge_channels(self, r_img, g_img, b_img, output_name="merged_image.jpg"):
        """Merge 3 grayscale images (channels) into 1 RGB image"""
        try:
            channels = (r_img.data, g_img.data, b_img.data)
            if any(channel is None or channel.ndim != 2 for channel in channels):
                raise ValueError("Mỗi kênh đầu vào phải là ảnh xám.")
            if len({channel.shape for channel in channels}) != 1:
                raise ValueError("Các kênh phải có cùng kích thước.")
            merged_data = np.dstack((r_img.data, g_img.data, b_img.data))
            iio.imwrite(output_name, merged_data)
            return Image(output_name, merged_data)
        except Exception as e:
            return None

    @classmethod
    def load_image_data(cls, path, source="opencv"):
        """Load one image as an RGB/grayscale ndarray using the selected backend."""
        images = cls().load_image(path, source=source)
        if not images:
            raise ValueError(f"Không thể đọc ảnh từ: {path}")
        return images[0].data


