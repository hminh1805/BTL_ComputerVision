import numpy as np
import math
import cv2

class Transform:
    def __init__(self):
        self.recolor = 0 # màu nền khi đổi ảnh (đen)
        return
    
    
    def translation(self, image,tx=0,ty=0):
        "dịch ảnh theo vector (tx,ty)"
        if image is None:
            raise ValueError("Ảnh đầu vào không hợp lệ.")
        
        #copy ảnh gốc sang ảnh output
        output = image.copy()
        #dịch phải
        if tx > 0:
            output[:,tx:,:] = output[:,:-tx:,:]
            output[:,:tx,:] = self.recolor
        if tx < 0:
            tx=-tx
            output[:,:-tx:,:] = output[:,tx:,:]
            output[:,-tx:,:] = self.recolor
        #dịch xuống
        if ty > 0:
            output[ty:,:,:] = output[:-ty:,:,:]
            output[:ty,:,:] = self.recolor
        if ty < 0:
            ty=-ty
            output[:-ty:,:,:] = output[ty:,:,:]
            output[-ty:,:,:] = self.recolor
        
        return output
    
    def rotation(self, image, angle):
        "xoay ảnh theo góc angle"
        if image is None:
            raise ValueError("Ảnh đầu vào không hợp lệ.")
        
        height, width, channels = image.shape
        angle_rad = math.radians(angle)
        output = image.copy()
        #tâm ảnh
        x_center = int(width / 2)
        y_center = int(height / 2)
        
        for y in range(height):
            for x in range(width):
                #kéo từng điểm về trung tâm ảnh
                centered_x = x - x_center
                centered_y = y - y_center
                
                #x' = x*cos(angle) - y*sin(angle)
                #y' = x*sin(angle) + y*cos(angle)
                new_x = int((centered_x)*math.cos(angle_rad) - (centered_y)*math.sin(angle_rad))
                new_y = int((centered_x)*math.sin(angle_rad) + (centered_y)*math.cos(angle_rad))
                
                #đưa điểm về vị trí cũ
                new_x += x_center
                new_y += y_center
                
                if 0<= new_x < width and 0<= new_y < height:
                    output[new_y,new_x,:] = image[y,x,:]
                else:
                    output[y,x,:] = self.recolor
                    
        return output
    
    def rotation_v2(self,image,angle):
        "xoay ảnh theo góc "
        if image is None:
            raise ValueError("Ảnh đầu vào không hợp lệ.")
        
        height, width, channels = image.shape
        angle_rad = math.radians(angle)
        output = image.copy()
        x_center = int(width / 2)
        y_center = int(height / 2)
        
        for y in range(height):
            for x in range(width):
                centered_x = x - x_center
                centered_y = y - y_center
                
                #x' = x*cos(angle) - y*sin(angle)
                #y' = x*sin(angle) + y*cos(angle)
                org_x = int(round(centered_x*math.cos(angle_rad) + centered_y*math.sin(angle_rad)))
                org_y = int(round(-centered_x*math.sin(angle_rad) + centered_y*math.cos(angle_rad)))
                
                if 0<= org_x + x_center < width and 0<= org_y + y_center < height:
                    output[y,x,:] = image[org_y + y_center,org_x + x_center,:]
                else:
                    output[y,x,:] = self.recolor
        return output
    
    def scaling(self,image,scale_x=1.0,scale_y=1.0):
        "phóng to/thu nhỏ ảnh theo tỉ lệ scale_x, scale_y"
        if image is None:
            raise ValueError("Ảnh đầu vào không hợp lệ.")
        
        height, width, channels = image.shape
        output = image.copy()
        x_center = int(width / 2)
        y_center = int(height / 2)
        
        for y in range(height):
            for x in range(width):
                #kéo về tâm ảnh
                centered_x = x - x_center
                centered_y = y - y_center
                
                org_x = int(round(centered_x/scale_x)) + x_center
                org_y = int(round(centered_y/scale_y)) + y_center
                
                if 0<= org_x < width and 0<= org_y < height:
                    output[y,x,:] = image[org_y,org_x,:]
                else:
                    output[y,x,:] = self.recolor
        return output
    
    def affine(self,image, src_points, dst_points):
        """
        Thực hiện Affine transformation dựa trên 3 cặp điểm.
        src_points: Tọa độ 3 điểm trên ảnh gốc. Ví dụ: np.float32([[50,50], [200,50], [50,200]])
        dst_points: Tọa độ 3 điểm đích mong muốn.
        """
        # 1. Dùng OpenCV để giải hệ 6 phương trình, tìm ra ma trận 2x3 chứa (a,b,c,d,e,f)
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # 2. Áp dụng ma trận biến đổi lên toàn bộ điểm ảnh
        height, width = image.shape[:2]
        output = cv2.warpAffine(image, affine_matrix, (width, height))
        
        return output

    def projective(self,image, src_points, dst_points):
        """
        Thực hiện Projective transformation dựa trên 4 cặp điểm.
        """
        # 1. Dùng OpenCV để giải hệ 8 phương trình, tìm ra ma trận 3x3 (có chứa g, h)
        projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 2. Áp dụng ma trận biến đổi phối cảnh
        height, width = image.shape[:2]
        output = cv2.warpPerspective(image, projective_matrix, (width, height))
        return output