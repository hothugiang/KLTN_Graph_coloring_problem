import pandas as pd

file_path = 'Res/input_stat.json' 

# Đọc dữ liệu từ file JSON
data = pd.read_json(file_path)

# Trích xuất kích thước đồ thị từ tên file
data['graph_size'] = data['graph'].str.extract(r'(\d+)_').astype(int)

# Nhóm dữ liệu theo kích thước đồ thị và tính toán thống kê
statistics = data.groupby('graph_size').agg(
    num_edges_mean=('num_edges', 'mean'),
    num_edges_std=('num_edges', 'std'),
    min_colors_mean=('min_colors', 'mean'),
    min_colors_std=('min_colors', 'std')
).reset_index()

# Hiển thị kết quả
print(statistics)
