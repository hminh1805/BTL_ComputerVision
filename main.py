from src.Transform import Transform
import numpy as np
from src.utils import load_image,show_image,rgb_to_gray

img = load_image("data/pixel.png")

# img = rgb_to_gray(img)

rows, cols = img.shape[:2]
transform = Transform()

# --- ORIGINAL IMAGE ---
print(img.shape)
show_image(img)

# --- TRANSLATION ---
res_img = transform.translation(img,10,10)
show_image(res_img)

# --- ROTATION ---
res_img = transform.rotation(img,45)
show_image(res_img)

# --- ROTATION V2 ---
res_img = transform.rotation_v2(img,45)
show_image(res_img)

#--- SCALING SMALL---
res_img = transform.scaling(img,0.5,0.5)
show_image(res_img)

#--- SCALING BIG---
res_img = transform.scaling(img,1.5,1.5)
show_image(res_img)

# ---  AFFINE ---
# 1. Chọn 3 điểm trên ảnh gốc
pts_source = np.array([
    [0, 0], 
    [cols - 1, 0], 
    [0, rows - 1],
], dtype=np.float32)

# 2. Định nghĩa 3 điểm đích
pts_dest = np.array([
    [0, 0],             
    [cols - 50, 0],     
    [50, rows - 1],
], dtype=np.float32)

res_img = transform.affine(img,src_points=pts_source,dst_points=pts_dest)
show_image(res_img)

# --- PROJECTIVE ---
# 1. Chọn 4 điểm trên ảnh gốc
pts_source = np.array([
    [0, 0], 
    [cols - 1, 0], 
    [0, rows - 1],
    [cols - 1, rows - 1]
], dtype=np.float32)

# 2. Định nghĩa 4 điểm đích
pts_dest = np.array([
    [0, 0],             
    [cols - 1, 0],     
    [50, rows - 1],
    [cols - 50, rows - 1]
], dtype=np.float32)

res_img = transform.projective(img,src_points=pts_source,dst_points=pts_dest)

show_image(res_img)
