# HỆ THỐNG DỰ ĐOÁN GIÁ CÁC MÃ CỔ PHIẾU CỦA VIỆT NAM

[![Streamlit](https://img.shields.io/badge/Go%20To-Streamlit%20Cloud-red?logo=streamlit)](https://polyps-segmentation-txgih92nkah.streamlit.app/)

## Tổng Quan

Dự án này bao gồm nhiều module để thu thập, tiền xử lý và phân tích dữ liệu chứng khoán và bài báo tin tức. Các module được thiết kế để lấy dữ liệu từ nhiều nguồn, tiền xử lý dữ liệu và thực hiện dự đoán giá chứng khoán.

## Cấu Trúc Thư Mục
├── app \
│ ├── dashboard.py \
│ └── stock_prediction_app.py \
├── crawler \
│ ├── newspaper_crawler.py \
│ ├── stock_crawler.py \
│ ├── stock_indice_crawler.py \
│ └── pre_processing.py \
├── data \
│ └── # Thư mục này chứa các tệp dữ liệu được thu thập và xử lý \
├── models \
│ ├── model_comparison_results.csv \
│ └── training_and_tuning_model.py \
├── README.md \
├── requirements.txt \

## Các Module

### 1. app
- **dashboard.py**: Tạo bảng điều khiển để trực quan hóa dữ liệu chứng khoán và dự đoán.
- **stock_prediction_app.py**: Ứng dụng dự đoán giá chứng khoán sử dụng các mô hình đã được huấn luyện.

### 2. crawler
- **newspaper_crawler.py**: Thu thập các bài báo từ các trang web tin tức.

- **stock_crawler.py**: Lấy dữ liệu lịch sử chứng khoán cho các mã cổ phiếu.
- **stock_indice_crawler.py**: Lấy dữ liệu lịch sử cho các chỉ số chứng khoán.
- **pre_processing.py**: Tiền xử lý dữ liệu thô thu được từ các trình thu thập dữ liệu.

### 3. data
- **# Thư mục này chứa các tệp dữ liệu được thu thập và xử lý.**

### 4. models
- **model_comparison_results.csv**: Kết quả so sánh các mô hình.
- **training_and_tuning_model.py**: Huấn luyện và tinh chỉnh các mô hình học máy để dự đoán giá chứng khoán.

## STARTED

### Điều Kiện Tiên Quyết

- Python 3.7+
- Các gói Python cần thiết (có thể cài đặt qua `requirements.txt`)

### Cài Đặt

1. Tải về và giải nén mã nguồn: https://drive.google.com/drive/folders/1Dk_TT5OUxHeXCDbcnpPijjBvFXinZybB?usp=sharing

2. Cài đặt các gói cần thiết
    ```sh
    pip install -r requirements.txt
    ```

### Sử Dụng

1. **Thu Thập Tin Tức**
    ```sh
    python crawler/newspaper_crawler.py
    ```

2. **Thu Thập Dữ Liệu Chứng Khoán**
    ```sh
    python crawler/stock_crawler.py
    ```

3. **Thu Thập Dữ Liệu Chỉ Số Chứng Khoán**
    ```sh
    python crawler/stock_indice_crawler.py
    ```

4. **Tiền Xử Lý Dữ Liệu**
    ```sh
    python crawler/pre_processing.py
    ```

5. **Huấn Luyện và Tinh Chỉnh Mô Hình**
    ```sh
    python models/training_and_tuning_model.py
    ```

6. **Chạy Ứng Dụng Dự Đoán Chứng Khoán**
    ```sh
    python app/stock_prediction_app.py
    ```

7. **Chạy Bảng Điều Khiển**
    ```sh
    python app/dashboard.py
    ```

## Cấu Hình

- Thay đổi danh sách mã cổ phiếu trong `stock_crawler.py` nếu cần.
- Thay đổi các URL và tham số trong `newspaper_crawler.py` để nhắm tới các trang web tin tức khác (các trang Vietnamtimes)
