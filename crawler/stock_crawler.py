import datetime
import os
import pandas as pd
from vnstock3 import *

if "ACCEPT_TC" not in os.environ:
    os.environ["ACCEPT_TC"] = "tôi đồng ý"

def main():
    stock_list = ["FPT", "VCB", "HPG", "VNM", "VIC"]

    # Nhập ngày bắt đầu và ngày kết thúc
    start_date = input("Nhập ngày bắt đầu (YYYY-MM-DD): ")
    end_date = input("Nhập ngày kết thúc (YYYY-MM-DD): ")
    
    # Kiểm tra định dạng ngày bắt đầu
    try:
        datetime.datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        print("Định dạng ngày bắt đầu không hợp lệ. Vui lòng nhập theo định dạng YYYY-MM-DD.")
        return
    
    # Kiểm tra định dạng ngày kết thúc
    try:
        datetime.datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        print("Định dạng ngày kết thúc không hợp lệ. Vui lòng nhập theo định dạng YYYY-MM-DD.")
        return

    for symbol in stock_list:
        print(f"Đang lấy dữ liệu cho mã cổ phiếu {symbol}...")
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        df = stock.quote.history(start=start_date, end=end_date, interval='1D')
        df['time'] = df['time'].dt.strftime('%Y-%m-%d')
        filename = f".//data//{symbol}_history.csv"
        df.to_csv(filename, index=False)
        print(f"Dữ liệu của mã cổ phiếu {symbol} đã được lưu vào file {filename}")

if __name__ == '__main__':
    main()
