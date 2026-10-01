import numpy as np
import cv2
from src.ImageLoader import Image
from src.utils import load_image, preprocess_image


class Filter:
    def __init__(self):
        pass

    def _preprocess_image(self, image):
        return preprocess_image(image, to_float32=True)

    def mean_kernel(self, kernel_size=3):
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size cua Mean filter phai la so le va >= 3 !!!")
        return np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)

    ## Ham de tang toc gaussian 2D
    def gaussian_kernel_1d(self, kernel_size=3, sigma=1.0):
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError("kernel_size cua Gaussian filter phai la so le va >= 3 !!!")
        radius = kernel_size // 2
        x = np.arange(-radius, radius + 1)
        g1d = np.exp(-(x ** 2) / (2.0 * (sigma ** 2)))
        return (g1d / np.sum(g1d)).astype(np.float32)

    def gaussian_kernel_2d(self, kernel_size=3, sigma=1.0):
        g1d = self.gaussian_kernel_1d(kernel_size, sigma)
        g2d = np.outer(g1d, g1d)
        return g2d.astype(np.float32)

    def sobel_kernel(self, axis="x", kernel_size=3):
        if kernel_size != 3:
            raise ValueError("Sobel kernel chuan phai co kich thuoc la 3x3 !!!")
        
        if axis == "x":
            return np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=np.float32)
        elif axis == "y":
            return np.array([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ], dtype=np.float32)
        else:
            raise ValueError("axis cua Sobel phai la 'x' hoac 'y' !!!")

    def laplacian_kernel(self, kernel_size=3, connectivity=4):
        if kernel_size != 3:
            raise ValueError("Laplacian kernel chuan phai co kich thuoc la 3x3 !!!")

        if connectivity == 4:
            return np.array([
                [ 0,  1,  0],
                [ 1, -4,  1],
                [ 0,  1,  0]
            ], dtype=np.float32)
        elif connectivity == 8:
            return np.array([
                [ 1,  1,  1],
                [ 1, -8,  1],
                [ 1,  1,  1]
            ], dtype=np.float32)
        else:
            raise ValueError("connectivity cua Laplacian 3x3 phai la 4 hoac 8 !!!")

    def convolution(self, image, kernel, mode="reflect"):
        img_arr = self._preprocess_image(image)

        if kernel.ndim != 2:
            raise ValueError("kernel phai la ma tran 2 chieu !!!")

        kh, kw = kernel.shape
        if kh % 2 == 0 or kw % 2 == 0:
            raise ValueError("Kich thuoc cua kernel phai la so le !!!")

        k_flipped = np.flip(kernel)
        pad_h = kh // 2
        pad_w = kw // 2

        ## Anh grey scale
        if img_arr.ndim == 2:
            h, w = img_arr.shape
            padded = np.pad(img_arr, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)
            output = np.zeros((h, w), dtype=np.float32)

            for u in range(kh):
                for v in range(kw):
                    weight = k_flipped[u, v]
                    if weight != 0:
                        output += weight * padded[u:u + h, v:v + w]
            return output

        ## Anh RGB
        elif img_arr.ndim == 3:
            h, w, c = img_arr.shape
            padded = np.pad(img_arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode=mode)
            output = np.zeros((h, w, c), dtype=np.float32)

            for u in range(kh):
                for v in range(kw):
                    weight = k_flipped[u, v]
                    if weight != 0:
                        output += weight * padded[u:u + h, v:v + w, :]
            return output

        else:
            raise ValueError("Anh phai la anh mau xam (2 chieu) hoac la anh mau RGB (3 chieu) !!!")

    def _separable_convolution(self, image, k1d_x, k1d_y, mode="reflect"):
        img_arr = self._preprocess_image(image)
        kx = k1d_x[::-1]
        ky = k1d_y[::-1]

        len_x = len(kx)
        len_y = len(ky)
        pad_x = len_x // 2
        pad_y = len_y // 2

        if img_arr.ndim == 2:
            h, w = img_arr.shape
            pad_horiz = np.pad(img_arr, ((0, 0), (pad_x, pad_x)), mode=mode)
            temp = np.zeros((h, w), dtype=np.float32)
            for v in range(len_x):
                temp += kx[v] * pad_horiz[:, v:v + w]

            pad_vert = np.pad(temp, ((pad_y, pad_y), (0, 0)), mode=mode)
            output = np.zeros((h, w), dtype=np.float32)
            for u in range(len_y):
                output += ky[u] * pad_vert[u:u + h, :]

            return output

        elif img_arr.ndim == 3:
            h, w, c = img_arr.shape
            pad_horiz = np.pad(img_arr, ((0, 0), (pad_x, pad_x), (0, 0)), mode=mode)
            temp = np.zeros((h, w, c), dtype=np.float32)
            for v in range(len_x):
                temp += kx[v] * pad_horiz[:, v:v + w, :]

            pad_vert = np.pad(temp, ((pad_y, pad_y), (0, 0), (0, 0)), mode=mode)
            output = np.zeros((h, w, c), dtype=np.float32)
            for u in range(len_y):
                output += ky[u] * pad_vert[u:u + h, :, :]

            return output



    def mean_filter(self, image, kernel_size=3):
        k = self.mean_kernel(kernel_size)
        res = self.convolution(image, k)
        return np.clip(res, 0, 255).astype(np.uint8)

    def gaussian_filter(self, image, kernel_size=3, sigma=1.0):
        # Đảm bảo kích thước kernel tối thiểu bao quát được phân phối của sigma
        calc_k = max(kernel_size, int(2 * np.ceil(2 * sigma) + 1)) | 1
        g1d = self.gaussian_kernel_1d(calc_k, sigma)
        res = self._separable_convolution(image, g1d, g1d)
        return np.clip(res, 0, 255).astype(np.uint8)

    def sobel_filter(self, image, axis="both", kernel_size=3):
        if kernel_size != 3:
            raise ValueError("Sobel filter chỉ hỗ trợ kích thước kernel 3x3!")

        if axis == "x":
            kx = self.sobel_kernel("x", kernel_size=3)
            gx = self.convolution(image, kx)
            return np.clip(np.abs(gx), 0, 255).astype(np.uint8)

        elif axis == "y":
            ky = self.sobel_kernel("y", kernel_size=3)
            gy = self.convolution(image, ky)
            return np.clip(np.abs(gy), 0, 255).astype(np.uint8)

        elif axis == "both":
            kx = self.sobel_kernel("x", kernel_size=3)
            ky = self.sobel_kernel("y", kernel_size=3)
            gx = self.convolution(image, kx)
            gy = self.convolution(image, ky)
            magnitude = np.sqrt(gx**2 + gy**2)
            return np.clip(magnitude, 0, 255).astype(np.uint8)

        else:
            raise ValueError("axis phải là 'x', 'y' hoặc 'both'!")

    def laplacian_filter(self, image, kernel_size=3, connectivity=4):
        if kernel_size != 3:
            raise ValueError("Laplacian filter chỉ hỗ trợ kích thước kernel 3x3!")

        k = self.laplacian_kernel(kernel_size=3, connectivity=connectivity)
        res = self.convolution(image, k)
        return np.clip(np.abs(res), 0, 255).astype(np.uint8)