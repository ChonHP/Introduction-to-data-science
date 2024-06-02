# File: app/stock_prediction_app.py
import os
from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Đặt style cho Matplotlib
plt.style.use('dark_background')

# Set up the Streamlit app
st.title("Dự đoán giá cổ phiếu Việt Nam")

# Chọn mã cổ phiếu
stock_symbol = st.selectbox("Chọn mã cổ phiếu", ["FPT", "HPG", "VCB", "VIC", "VNM"])

# Chọn mô hình
model_files = os.listdir("models")
model_files = [f for f in model_files if stock_symbol in f]
model_name = st.selectbox("Chọn mô hình", model_files)

# Chọn số ngày muốn dự đoán
days_to_predict = st.number_input("Số ngày muốn dự đoán", min_value=1, max_value=30, value=7)

# Tải dữ liệu
data_path = f"data/merged_{stock_symbol}_data.csv"
data = pd.read_csv(data_path)

# Đổi tên cột 'Unnamed: 0' thành 'Date' nếu nó tồn tại
if 'Unnamed: 0' in data.columns:
    data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

# Đảm bảo cột 'Date' được chuyển sang định dạng datetime và đặt làm chỉ mục
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Tải mô hình đã chọn
model_path = os.path.join("models", model_name)
model = joblib.load(model_path)

# Chuẩn bị dữ liệu đặc trưng
target_column = f'{stock_symbol}_close'
X = data.drop(columns=[target_column])
y = data[target_column]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dự đoán giá trong tương lai
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, days_to_predict + 1)]
future_X = X_scaled[-days_to_predict:]

# Đảm bảo future_X có cấu trúc giống nhau
if future_X.shape[0] < days_to_predict:
    last_row = future_X[-1]
    rows_to_add = days_to_predict - future_X.shape[0]
    future_X = np.vstack([future_X, np.tile(last_row, (rows_to_add, 1))])

# Thực hiện dự đoán
future_prices = model.predict(future_X)

# Kiểm tra giá trị dự đoán âm
if any(future_prices < 0):
    st.warning("Một số giá trị dự đoán là âm. Hãy kiểm tra lại dữ liệu và mô hình.")
else:
    st.success("Giá trị dự đoán hợp lệ.")

# Trực quan hóa kết quả dự đoán
fig, ax = plt.subplots()
ax.plot(future_dates, future_prices, label='Giá dự đoán', linestyle='--', color='cyan')
ax.set_xlabel('Ngày')
ax.set_ylabel('Giá')
ax.set_title(f'Dự đoán giá cổ phiếu {stock_symbol}', color='white')
ax.legend()
fig.set_size_inches(10, 6)

# Hiển thị kết quả trên Streamlit
st.write(f"Giá dự đoán cho cổ phiếu {stock_symbol} trong {days_to_predict} ngày tới:")
st.write(pd.DataFrame({'Ngày': future_dates, 'Giá dự đoán': future_prices}))

st.pyplot(fig)

# Biểu đồ thêm cho nhà đầu tư
# Biểu đồ histogram của giá dự đoán
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(future_prices, bins=20, edgecolor='white', color='magenta')
ax_hist.set_xlabel('Giá dự đoán', color='white')
ax_hist.set_ylabel('Tần suất', color='white')
ax_hist.set_title('Phân bố giá dự đoán', color='white')
fig_hist.set_size_inches(10, 6)

st.pyplot(fig_hist)

# Biểu đồ đường của biến động giá dự đoán
fig_trend, ax_trend = plt.subplots()
ax_trend.plot(future_dates, np.diff(future_prices, prepend=future_prices[0]), label='Biến động giá', color='lime')
ax_trend.set_xlabel('Ngày', color='white')
ax_trend.set_ylabel('Biến động giá', color='white')
ax_trend.set_title('Biến động giá dự đoán qua các ngày', color='white')
ax_trend.legend()
fig_trend.set_size_inches(10, 6)

st.pyplot(fig_trend)
