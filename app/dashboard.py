import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load the combined data from multiple CSV files
fpt_data = pd.read_csv('data//merged_FPT_data.csv')
hpg_data = pd.read_csv('data//merged_HPG_data.csv')
vcb_data = pd.read_csv('data//merged_VCB_data.csv')
vic_data = pd.read_csv('data//merged_VIC_data.csv')
vnm_data = pd.read_csv('data//merged_VNM_data.csv')

# Add a Symbol column to each dataframe
fpt_data['Symbol'] = 'FPT'
hpg_data['Symbol'] = 'HPG'
vcb_data['Symbol'] = 'VCB'
vic_data['Symbol'] = 'VIC'
vnm_data['Symbol'] = 'VNM'

# Combine the data
combined_data = pd.concat([fpt_data, hpg_data, vcb_data, vic_data, vnm_data], ignore_index=True)

# Ensure 'Date' is in the correct datetime format for the charts to display correctly
combined_data['Unnamed: 0'] = pd.to_datetime(combined_data['Unnamed: 0'])
combined_data = combined_data.rename(columns={'Unnamed: 0': 'Date'})

# Create the Dash application
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Phân Tích Cảm Xúc và Giá Cổ Phiếu ETF'),
    html.Div(children='Lựa chọn mã cổ phiếu:'),
    dcc.Dropdown(
        id='stock-symbol-dropdown',
        options=[{'label': symbol, 'value': symbol} for symbol in combined_data['Symbol'].unique()],
        value=combined_data['Symbol'].unique()[0],
        multi=False
    ),
    html.Div(children='Biểu đồ khối lượng giao dịch theo thời gian.'),
    dcc.Graph(id='volume-time-series'),
    html.Div(children='Biểu đồ giá đóng cửa theo thời gian.'),
    dcc.Graph(id='close-price-time-series'),
    html.Div(children='Biểu đồ tương quan giữa điểm số cảm xúc và giá đóng cửa cổ phiếu.'),
    dcc.Graph(id='correlation-heatmap'),
    html.Div(children='Biểu đồ các chỉ số chứng khoán.'),
    dcc.Graph(id='stock-indexes'),
    html.Div(children='Biểu đồ phân tán giữa giá đóng cửa và điểm số cảm xúc.'),
    dcc.Graph(id='scatter-price-sentiment'),
    html.Div(children='Biểu đồ phân tán giữa các chỉ số chứng khoán với giá đóng cửa.'),
    dcc.Graph(id='scatter-indexes-price'),
    html.Div(children='Biểu đồ phân tán giữa các chỉ số chứng khoán với điểm số cảm xúc.'),
    dcc.Graph(id='scatter-indexes-sentiment'),
    html.Div(children='Biểu đồ tương quan giữa các chỉ số chứng khoán với giá đóng cửa.'),
    dcc.Graph(id='correlation-indexes-price'),
    html.Div(children='Biểu đồ tương quan giữa các chỉ số chứng khoán với điểm số cảm xúc.'),
    dcc.Graph(id='correlation-indexes-sentiment')
])

# Callbacks for updating the graphs
@app.callback(
    Output('volume-time-series', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_volume_graph(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    fig = px.line(filtered_data, x='Date', y=f'{selected_symbol}_volume', title=f'Khối Lượng Giao Dịch Của {selected_symbol} Theo Thời Gian')
    return fig

@app.callback(
    Output('close-price-time-series', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_close_price_graph(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    fig = px.line(filtered_data, x='Date', y=f'{selected_symbol}_close', title=f'Giá Đóng Cửa Của {selected_symbol} Theo Thời Gian')
    return fig

@app.callback(
    Output('correlation-heatmap', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_correlation_heatmap(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    correlation = filtered_data[[f'{selected_symbol}_close', 'sentiment_score']].corr()
    fig = px.imshow(correlation, text_auto=True, title=f'Tương Quan Giữa Giá và Điểm Số Cảm Xúc Của {selected_symbol}')
    return fig

@app.callback(
    Output('stock-indexes', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_stock_indexes(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    fig = px.line(filtered_data, x='Date', 
                  y=['Dollar_Index_Close', 'Dow_Jones_Close', 'Nasdaq_Close', 'US_30_Close', 'US_500_Close'], 
                  title=f'Các Chỉ Số Chứng Khoán Theo Thời Gian')
    return fig

@app.callback(
    Output('scatter-price-sentiment', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_scatter_price_sentiment(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    fig = px.scatter(filtered_data, x='sentiment_score', y=f'{selected_symbol}_close', color=f'{selected_symbol}_close', 
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title=f'Biểu Đồ Phân Tán Giữa Giá Đóng Cửa và Điểm Số Cảm Xúc Của {selected_symbol}')
    return fig

@app.callback(
    Output('scatter-indexes-price', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_scatter_indexes_price(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    fig = px.scatter_matrix(filtered_data, dimensions=['Dollar_Index_Close', 'Dow_Jones_Close', 'Nasdaq_Close', 'US_30_Close', 'US_500_Close', f'{selected_symbol}_close'],
                            title=f'Biểu Đồ Phân Tán Giữa Các Chỉ Số Chứng Khoán và Giá Đóng Cửa Của {selected_symbol}')
    return fig

@app.callback(
    Output('scatter-indexes-sentiment', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_scatter_indexes_sentiment(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    fig = px.scatter_matrix(filtered_data, dimensions=['Dollar_Index_Close', 'Dow_Jones_Close', 'Nasdaq_Close', 'US_30_Close', 'US_500_Close', 'sentiment_score'],
                            title=f'Biểu Đồ Phân Tán Giữa Các Chỉ Số Chứng Khoán và Điểm Số Cảm Xúc Của {selected_symbol}')
    return fig

@app.callback(
    Output('correlation-indexes-price', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_correlation_indexes_price(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    correlation = filtered_data[['Dollar_Index_Close', 'Dow_Jones_Close', 'Nasdaq_Close', 'US_30_Close', 'US_500_Close', f'{selected_symbol}_close']].corr()
    fig = px.imshow(correlation, text_auto=True, title=f'Tương Quan Giữa Các Chỉ Số Chứng Khoán và Giá Đóng Cửa Của {selected_symbol}')
    return fig

@app.callback(
    Output('correlation-indexes-sentiment', 'figure'),
    Input('stock-symbol-dropdown', 'value')
)
def update_correlation_indexes_sentiment(selected_symbol):
    filtered_data = combined_data[combined_data['Symbol'] == selected_symbol]
    correlation = filtered_data[['Dollar_Index_Close', 'Dow_Jones_Close', 'Nasdaq_Close', 'US_30_Close', 'US_500_Close', 'sentiment_score']].corr()
    fig = px.imshow(correlation, text_auto=True, title=f'Tương Quan Giữa Các Chỉ Số Chứng Khoán và Điểm Số Cảm Xúc Của {selected_symbol}')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)