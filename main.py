from src.Transform import Transform
import numpy as np
from src.utils import load_image,show_image,rgb_to_gray
from src.ImageLoader import ImageLoader

il = ImageLoader()
image = il.load_image("data/pixel.png")
img = image.data if image is not None else None


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


r,g,b = il.split_channels(img)
r = il.gray_to_rgb(r)
g = il.gray_to_rgb(g)
b = il.gray_to_rgb(b)
show_image(r)
show_image(g)
show_image(b)

r,g,b = il.split_channels(img)

merged_img = il.merge_channels(r,g,b)
show_image(merged_img)

merged_a = il.merge_channels(r,b,g)
show_image(merged_a)

merged_a = il.merge_channels(g,g,g)
show_image(merged_a)

merged_a = il.merge_channels(g,b,b)
show_image(merged_a)