# ⬡ Computer Vision

Đây là dự án môn học Xử lý ảnh (Computer Vision) tại Trường Đại học Bách Khoa - ĐHQG TP.HCM (HCMUT). 

Dự án cung cấp một giao diện đồ họa (GUI) trực quan để thực nghiệm các thuật toán xử lý ảnh cơ bản. Hầu hết các thuật toán cốt lõi đều được hiện thực từ đầu (from scratch) bằng Python và NumPy nhằm mục đích hiểu rõ bản chất toán học của ma trận ảnh, hạn chế tối đa việc phụ thuộc vào các hàm đóng gói sẵn (black-box) của thư viện chuyên dụng.

## ✨ Các tính năng chính

Dự án được chia thành 3 mô-đun thực nghiệm chính:

1. **🎨 Biểu diễn ảnh màu (Color Representation)**
   - Chuyển đổi qua lại giữa không gian màu RGB và Grayscale.
   - Bóc tách các kênh màu độc lập (R, G, B) thành các ma trận riêng biệt.
   - Hoán đổi vị trí các kênh màu (ví dụ: RGB sang BGR).

2. **🔍 Lọc không gian (Spatial Filtering)**
   - Tự xây dựng cơ chế nhân chập (Convolution) xử lý viền ảnh (padding).
   - **Bộ lọc thông thấp (Khử nhiễu/Làm mượt):** Mean Filter, Gaussian Filter (hỗ trợ tùy chỉnh `kernel_size` và `sigma`).
   - **Bộ lọc thông cao (Dò biên):** Sobel (X, Y, Magnitude), Laplacian (4-connectivity, 8-connectivity).

3. **🔄 Biến đổi hình học (Geometric Transformation)**
   - Các phép biến đổi tuyến tính: Tịnh tiến (Translation), Thu phóng (Scaling), Xoay (Rotation).
   - Tích hợp chiến lược **Ánh xạ ngược (Inverse Mapping)** để khắc phục lỗi rỗng điểm ảnh (holes) khi xoay và thu phóng.
   - Các phép biến đổi nâng cao: Affine (bảo toàn tính song song) và Projective/Homography (mô hình hóa phối cảnh).

## 📂 Cấu trúc thư mục

```text
├── data/                 # Thư mục chứa các ảnh mẫu dùng để kiểm thử
├── src/                  # Thư mục chứa mã nguồn cốt lõi
│   ├── ImageLoader.py    # Xử lý đọc/ghi và biểu diễn không gian màu
│   ├── Filter.py         # Hiện thực nhân chập và các bộ lọc không gian
│   ├── Transform.py      # Hiện thực các phép biến đổi hình học
│   └── utils.py          # Các hàm tiện ích hỗ trợ
├── test_sample/          # Thư mục lưu trữ kết quả ảnh đầu ra
├── app.py                # Tệp thực thi giao diện đồ họa (GUI) bằng Tkinter
├── main.py               # Tệp kịch bản chạy toàn bộ thực nghiệm (CLI)
└── requirements.txt      # Danh sách các thư viện phụ thuộc
```
🚀 **Hướng dẫn cài đặt**
Khuyến nghị sử dụng môi trường ảo (Virtual Environment) để tránh xung đột thư viện.


💻 **Hướng dẫn sử dụng**
Để trải nghiệm trực quan nhất, hãy khởi chạy ứng dụng GUI:

Bash
python app.py
Thao tác trên giao diện:

Cửa sổ ứng dụng hiện lên với 3 Tab chức năng (Màu sắc, Lọc ảnh, Biến đổi).

Tại phần Đầu vào (cột bên trái), nhấn nút 📄 Ảnh để nạp hình ảnh từ máy tính (có thể chọn thư mục data/ có sẵn của dự án).

Chọn phép toán và tinh chỉnh tham số tương ứng.

Nhấn Thực hiện để xem kết quả trực tiếp trên màn hình (đối chiếu ngay cạnh ảnh gốc).
