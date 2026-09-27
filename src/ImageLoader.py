import os
import imageio.v3 as iio
from skimage import color
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import math
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
    def __init__(self):
        self.image_list = []

    def _process_single_file(self, file_path):
        """Internal helper to read a single image file"""
        try:
            img_data = iio.imread(file_path)
            
            # Đảm bảo dữ liệu luôn ở chuẩn 8-bit (0-255)
            if img_data.dtype != np.uint8:
                img_data = img_as_ubyte(img_data)
                 
            return Image(file_path, img_data)
        except Exception as e:
            print(f"Error loading '{file_path}': {e}")
            return None

    def load_image(self, path):
        self.image_list = []
        
        if os.path.isfile(path):
            file_type, _ = mimetypes.guess_type(path)
            if file_type and file_type.startswith('image/'):
                img_obj = self._process_single_file(path)
                if img_obj: self.image_list.append(img_obj)
            else:
                print(f"Bỏ qua file không phải ảnh: {path}")
            
            return self.image_list

        elif os.path.isdir(path):
            # Quét toàn bộ file trong thư mục
            for name in os.listdir(path):
                path_file = os.path.join(path, name)
                
                # Bỏ qua nếu là thư mục con
                if not os.path.isfile(path_file):
                    continue
                    
                # Tự động đoán loại file
                file_type, _ = mimetypes.guess_type(path_file)
                if file_type and file_type.startswith('image/'):
                    img_obj = self._process_single_file(path_file)
                    if img_obj:
                        self.image_list.append(img_obj)
                        
            print(f"Đã load thành công {len(self.image_list)} ảnh.")
            return self.image_list
            
        else:
            print("Đường dẫn không hợp lệ!")
            return []

    def rgb_to_gray(self, img_obj):
        """Convert a color image to grayscale"""
        try:
            gray_data = img_as_ubyte(color.rgb2gray(img_obj.data))
            return Image(f"gray_{os.path.basename(img_obj.path)}", gray_data)
        except Exception as e:
            print(f"Error in rgb_to_gray: {e}")
            return None

    
