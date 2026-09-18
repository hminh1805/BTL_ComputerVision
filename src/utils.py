from ImageLoader import ImageLoader
import cv2
import numpy as np

image_loader = ImageLoader()

def load_image(image_path):
    # img = image_loader.load_image(image_path)

    # if img is None:
    #     img = cv2.imread(image_path)
    #     if img is None:
    #         raise ValueError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không tìm thấy ảnh tại đường dẫn: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
    
def rgb_to_gray(image: np.ndarray):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)