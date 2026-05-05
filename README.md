# Hướng dẫn chạy project Naive Bayes spam filter

## 1. Giới thiệu

Dự án này xây dựng mô hình Naive Bayes đơn giản để phát hiện email spam/ham. Dữ liệu được đọc từ file CSV trong thư mục `data_preprocessing`, tiền xử lý bằng `underthesea` và mô hình được lưu vào file `spam_model.json`.

## 2. Chuẩn bị môi trường

### 2.1. Kích hoạt virtualenv

Dự án đã có sẵn thư mục `env/` chứa môi trường ảo Python. Bạn nên kích hoạt môi trường trước khi chạy.

Trên Linux/macOS:

```bash
source env/bin/activate
```

### 2.2. Cài đặt thư viện

Nếu môi trường ảo chưa có đầy đủ thư viện, bạn có thể cài thêm bằng:

```bash
pip install underthesea
```

Nếu bạn muốn cài toàn bộ phụ thuộc từ file `requirements.txt` mà không có sẵn, hãy tạo file này hoặc cài trực tiếp các gói cần thiết.

## 3. Cấu trúc file chính

- `main.py`: mã nguồn chính để huấn luyện và dự đoán email spam.
- `spam_model.json`: file lưu mô hình đã huấn luyện.
- `data_preprocessing/final_huge_spam_dataset.csv`: dữ liệu tiếng Anh.
- `data_preprocessing/multilingual_huge_dataset.csv`: dữ liệu đa ngôn ngữ.

## 4. Chạy project

Từ thư mục gốc `bayes`, dùng lệnh:

```bash
python main.py
```

Khi chạy lần đầu, nếu `spam_model.json` chưa tồn tại, chương trình sẽ:

- đọc dữ liệu từ `final_huge_spam_dataset.csv` và `multilingual_huge_dataset.csv`
- huấn luyện mô hình Naive Bayes
- đánh giá trên tập test
- lưu mô hình vào `spam_model.json`

Nếu `spam_model.json` đã tồn tại, chương trình sẽ tải mô hình từ file và chuyển thẳng sang chế độ test thủ công.

## 5. Chế độ test thủ công

Sau khi chạy `main.py`, chương trình sẽ vào chế độ test thủ công. Bạn nhập câu email và nhận kết quả:

- `spam` → email rác
- `ham` → email bình thường

Gõ `quit`, `exit` hoặc `q` để thoát.

## 6. Ghi chú

- Dự án hiện tiền xử lý văn bản với `underthesea` và biểu diễn token bằng khoảng trắng.
- Nếu cần huấn luyện lại với dữ liệu khác, thay đổi đường dẫn file trong `main.py` và chạy lại.
- Đảm bảo dữ liệu CSV có ít nhất 2 cột: nhãn ở cột đầu và nội dung email ở cột thứ hai.
