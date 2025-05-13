import pandas as pd
import numpy as np

# Đọc dữ liệu từ file CSV
df = pd.read_csv("Iot/human_vital_signs_dataset_2024.csv")

# Lấy 2 cột cần thiết
df_selected = df[['Heart Rate', 'Oxygen Saturation']].copy()

# Hàm gán nhãn: 0 = bình thường, 1 = bất thường
def classify_label(row):
    if 60 <= row['Heart Rate'] <= 100 and row['Oxygen Saturation'] >= 95:
        return 0
    else:
        return 1

# Gán nhãn
df_selected['Label'] = df_selected.apply(classify_label, axis=1)

# Lọc ra chỉ các mẫu bình thường (Label = 0)
df_label_0 = df_selected[df_selected['Label'] == 0]

# Lấy ngẫu nhiên 10.000 mẫu
df_sampled = df_label_0.sample(n=10000, random_state=42)

# Shuffle lại lần nữa (nếu muốn)
df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Lưu ra file CSV
df_sampled.to_csv("Iot/heart_rate_oxygen_label_0.csv", index=False)
