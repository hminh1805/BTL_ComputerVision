# Computer Vision Project 1

Dự án môn học Xử lý ảnh (Computer Vision). Các code đầy đủ cần được cung cấp qua repository này ở chế độ public access.

## 📂 Cấu trúc thư mục
- `data/`: Chứa ảnh đầu vào (không push lên git). Tự tạo thư mục này ở local.
- `src/`: Chứa mã nguồn Python (`.py`).
  - `utils.py`: Chứa các hàm load,convert ảnh dùng chung. Mọi người import từ đây.
- `notebooks/`: Chứa file `.ipynb` để chạy demo, hiển thị kết quả và xuất báo cáo.

## 📌 TODO List & Phân công công việc

Quy trình làm việc:
1. Tạo branch mới từ `main` (VD: `feature/part1`).
2. Code và test trên branch của mình.
3. Tạo Pull Request để review trước khi gộp vào `main`.

### 🧑‍💻 Người 1: Biểu diễn ảnh màu và ảnh xám
Mục tiêu: Hiểu mối quan hệ giữa ảnh màu và ảnh xám, tách kênh màu.
- [ ] Viết vào file `src/ImageLoader.py` .
- [ ] Viết hàm `load_image` chuyển đổi ảnh màu sang ảnh xám mà **không dùng hàm cv2 có sẵn**
- [ ] Viết hàm `rgb_to_gray` chuyển đổi ảnh màu sang ảnh xám mà **không dùng hàm cv2 có sẵn**
- [ ] Viết hàm tách riêng từng kênh màu (R, G, B) từ ảnh gốc và biểu diễn dưới dạng ảnh xám
- [ ] Viết hàm kết hợp các kênh màu để tái tạo ảnh ban đầu và tạo ảnh màu mới (hoán đổi/thay thế kênh)
- [ ] **Báo cáo:** Giải thích rõ nguyên lý chuyển đổi, nhận xét sự khác biệt trực quan khi thiếu thông tin kênh màu

### 🧑‍💻 Người 2: Lọc ảnh với Low-pass và High-pass Filter
Mục tiêu: Hiểu bản chất các phép lọc trong không gian
*Lưu ý: Dùng hàm trong `utils.py` để lấy ảnh.*
- [ ] Viết vào file `src/Filter.py`.
- [ ] Implement Low-pass filtering (làm trơn, giảm nhiễu) bằng kernel chuẩn (mean, Gaussian) hoặc tự thiết kế
- [ ] Implement High-pass filtering (làm nổi biên, chi tiết) bằng kernel chuẩn (Laplacian, Sobel) hoặc tự thiết kế
- [ ] Chạy test các phép lọc trên ít nhất 2-3 ảnh khác nhau
- [ ] **Báo cáo:** Trình bày kernel sử dụng, phân tích sự khác biệt khi áp dụng trên các ảnh khác nhau (nhiều/ít chi tiết, có nhiễu)

### 🧑‍💻 Người 3: Biến đổi hình học (Transformations)
Mục tiêu: Xây dựng demo và phân biệt Affine vs Projective
- [ ] Tạo file `src/visualize.py`.
- [ ] Viết code minh họa các phép biến đổi: Translation, Rotation, Scaling
- [ ] Viết code minh họa Affine transformation trên cùng một ảnh đầu vào
- [ ] Viết code minh họa Projective (homography) transformation trên cùng một ảnh đầu vào
- [ ] **Báo cáo:** So sánh trực quan sự khác nhau giữa Affine và Projective, thảo luận về hạn chế và trường hợp sử dụng phù hợp của từng phương pháp

## 📝 Tổng hợp báo cáo (Cả nhóm)
- [ ] Tập hợp các hình ảnh kết quả từ `notebooks/`.
- [ ] Rà soát chéo: Nội dung cần thể hiện sự hiểu đúng bản chất, không chỉ trình bày kết quả
- [ ] Đưa các đoạn code chính, ngắn gọn vào báo cáo Word/PDF