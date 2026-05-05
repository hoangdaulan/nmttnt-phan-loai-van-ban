# Hướng dẫn chạy project Naive Bayes Spam Filter

## 1. Giới thiệu

Dự án này xây dựng mô hình Naive Bayes để phát hiện email spam/ham. Dữ liệu được đọc từ file CSV trong thư mục `data_preprocessing`, tiền xử lý bằng thư viện `underthesea` và mô hình được lưu vào file `spam_model.json`.

## 2. Chuẩn bị môi trường

### 2.1 Linux/macOS

#### Bước 1: Tạo virtual environment

```bash
python3 -m venv env
```

#### Bước 2: Kích hoạt virtual environment

```bash
source env/bin/activate
```

#### Bước 3: Cài đặt thư viện

```bash
pip install --upgrade pip
pip install underthesea
```

### 2.2 Windows

#### Bước 1: Tạo virtual environment

```bash
python -m venv env
```

#### Bước 2: Kích hoạt virtual environment

```bash
env\Scripts\activate
```

#### Bước 3: Cài đặt thư viện

```bash
pip install --upgrade pip
pip install underthesea
```

## 3. Cấu trúc file chính

- `main.py`: mã nguồn chính để huấn luyện và dự đoán email spam
- `spam_model.json`: file lưu mô hình đã huấn luyện (được tạo tự động)
- `data_preprocessing/final_huge_spam_dataset.csv`: dữ liệu tiếng Anh
- `data_preprocessing/multilingual_huge_dataset.csv`: dữ liệu đa ngôn ngữ
- `emails.csv`: dữ liệu mẫu để test

## 4. Chạy project

### Lần chạy đầu tiên (Huấn luyện mô hình)

```bash
python main.py
```

Chương trình sẽ:

- Đọc dữ liệu từ `final_huge_spam_dataset.csv` (tối đa 5000 email tiếng Anh)
- Đọc dữ liệu từ `multilingual_huge_dataset.csv` (tối đa 5000 email đa ngôn ngữ)
- Chia dữ liệu thành tập train (80%) và test (20%)
- Huấn luyện mô hình Naive Bayes
- Đánh giá mô hình trên tập test (in Accuracy, Precision, Recall, F1-Score)
- Lưu mô hình vào `spam_model.json`
- Chuyển sang chế độ test thủ công

**Lưu ý:** Lần chạy đầu tiên sẽ mất một số thời gian để xử lý dữ liệu và huấn luyện.

### Các lần chạy sau

```bash
python main.py
```

Chương trình sẽ:

- Phát hiện `spam_model.json` đã tồn tại
- Tải mô hình từ file
- Nhảy thẳng đến chế độ test thủ công

## 5. Chế độ test thủ công

Sau khi chạy `python main.py`, bạn sẽ vào chế độ test thủ công:

```
========================================
CHẾ ĐỘ TEST THỦ CÔNG (Gõ 'quit' để thoát)
========================================

Nhập nội dung email tiếng Anh:
```

Nhập nội dung email bất kỳ, chương trình sẽ dự đoán:

- `🔴 ĐÂY LÀ EMAIL RÁC (SPAM)` → email spam
- `🟢 EMAIL BÌNH THƯỜNG (HAM)` → email bình thường

**Thoát:** Gõ `quit`, `exit` hoặc `q` rồi nhấn Enter

## 6. Ghi chú

- **Tiền xử lý:** Dự án sử dụng `underthesea` để tách từ tiếng Việt và tiếng Anh
- **Laplace smoothing:** Sử dụng để tránh xác suất bằng 0 cho từ chưa từng xuất hiện
- **Huấn luyện lại:** Để huấn luyện với dữ liệu khác, xóa `spam_model.json` rồi chạy lại
- **Định dạng CSV:** File dữ liệu phải có ít nhất 2 cột: cột 1 là nhãn (0/ham hoặc 1/spam), cột 2 là nội dung email

## 7. Xóa mô hình (reset)

Để huấn luyện lại mô hình từ đầu:

```bash
rm spam_model.json
python main.py
```

Hoặc trên Windows:

```bash
del spam_model.json
python main.py
```
