import yfinance as yf
import pandas as pd
import datetime

def get_index_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def main():
    # Danh sách các chỉ số chứng khoán cần thu thập
    indices = {
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "US 30": "YM=F",
        "US 500": "ES=F",
        "Dollar Index": "DX-Y.NYB"
    }
    
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

    # Lấy dữ liệu và lưu vào CSV
    for name, ticker in indices.items():
        print(f"Đang lấy dữ liệu cho chỉ số {name} ({ticker})...")
        data = get_index_data(ticker, start_date, end_date)
        filename = f".//data//{name.replace(' ', '_')}_history.csv"
        data.to_csv(filename)
        print(f"Dữ liệu của chỉ số {name} đã được lưu vào file {filename}")

if __name__ == '__main__':
    main()
