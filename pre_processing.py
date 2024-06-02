import pandas as pd
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Tải dữ liệu
file_path = 'data//article.csv'
data = pd.read_csv(file_path)

# Đặt tên cột phù hợp
data.columns = ['Title', 'Date', 'Content']

# Chia cột Date thành hai phần Date và Time
split_date_time = data['Date'].str.split(' \| ', expand=True)
data[['Date', 'Time']] = split_date_time

# Chuyển đổi cột Date sang định dạng datetime
data['Date'] = pd.to_datetime(data['Date'], format='%b %d, %Y', errors='coerce')

# Loại bỏ cột Time vì không cần thiết cho phân tích này
data.drop(columns=['Time'], inplace=True)

# Khởi tạo bộ phân tích cảm xúc VADER
analyzer = SentimentIntensityAnalyzer()

# Tính toán sentiment scores cho cột Content
data['sentiment_score'] = data['Content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Lưu lại dữ liệu đã tiền xử lý vào tệp CSV mới
output_file_path = 'data//preprocessed_articles_with_sentiment.csv'
data.to_csv(output_file_path, index=False)

# Hiển thị thông báo hoàn thành
print(f"Dữ liệu đã được lưu vào tệp: {output_file_path}")

# Đường dẫn đến các file dữ liệu
file_paths = [
    ('Dollar_Index', 'data//Dollar_Index_history.csv'),
    ('Dow_Jones', 'data//Dow_Jones_history.csv'),
    ('FPT', 'data//FPT_history.csv'),
    ('HPG', 'data//HPG_history.csv'),
    ('Nasdaq', 'data//Nasdaq_history.csv'),
    ('US_30', 'data//US_30_history.csv'),
    ('US_500', 'data//US_500_history.csv'),
    ('VCB', 'data//VCB_history.csv'),
    ('VIC', 'data//VIC_history.csv'),
    ('VNM', 'data//VNM_history.csv')
]

# Đọc dữ liệu từ các file CSV
dataframes = {}
for name, path in file_paths:
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

    # Xóa cột Dollar_Index_Volume nếu tồn tại
    if 'Volume' in df.columns:
        df.drop(columns=['Volume'], inplace=True)

    dataframes[name] = df
    print(f"Dataframe {name} after setting datetime index:")
    print(df.head())


def fill_missing_values(df):
    df = df.copy()
    # Điền các giá trị khuyết bằng trung bình của cột tương ứng
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
    return df


# Chuyển đổi dữ liệu time-series thành dữ liệu đặc trưng
def create_features(df, label):
    df = df.copy()
    close_col = 'Close' if 'Close' in df.columns else 'close'
    df['return'] = df[close_col].pct_change()
    df['ma5'] = df[close_col].rolling(window=5).mean()
    df['ma10'] = df[close_col].rolling(window=10).mean()
    df['std_dev'] = df[close_col].rolling(window=10).std()
    df['ema10'] = df[close_col].ewm(span=10, adjust=False).mean()

    # Điền giá trị khuyết sử dụng phương pháp trung bình động
    df = fill_missing_values(df)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    feature_cols = ['return', 'ma5', 'ma10', 'std_dev', 'ema10']
    df.dropna(inplace=True)  # Đảm bảo không có giá trị NA trước khi chuẩn hóa
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.columns = [f"{label}_{col}" for col in df.columns]  # Nhãn hóa tên cột

    print(f"Features for {label} after creation and normalization:")
    print(df.head())

    return df


# Đọc dữ liệu bài báo đã tiền xử lý
articles_path = 'data//preprocessed_articles_with_sentiment.csv'
articles_data = pd.read_csv(articles_path)
articles_data['Date'] = pd.to_datetime(articles_data['Date'])
articles_data.set_index('Date', inplace=True)

# Kết hợp dữ liệu bài báo với dữ liệu chứng khoán
stocks = ['FPT', 'HPG', 'VCB', 'VIC', 'VNM']
indexes = ['Dollar_Index', 'Dow_Jones', 'Nasdaq', 'US_30', 'US_500']

for stock in stocks:
    merged_df = create_features(dataframes[stock], stock)
    for index in indexes:
        if index in dataframes:
            features_df = create_features(dataframes[index], index)
            merged_df = merged_df.join(features_df, how='inner')

    # Kết hợp với dữ liệu bài báo
    merged_df = merged_df.join(articles_data['sentiment_score'], how='left')

    # Điền giá trị khuyết sử dụng phương pháp trung bình động
    merged_df = fill_missing_values(merged_df)

    output_path = f'data//merged_{stock}_data.csv'
    merged_df.to_csv(output_path)
    print(f"Data processing complete for {stock}. The merged data has been saved to {output_path}")
    print(merged_df.head())  # In một vài hàng đầu tiên của DataFrame đã kết hợp cuối cùng
