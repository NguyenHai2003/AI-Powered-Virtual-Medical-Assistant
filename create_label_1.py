import pandas as pd
import numpy as np

# Số mẫu cần tạo
n_samples = 10000
n_each = n_samples // 3

np.random.seed(42)

# Trường hợp 1: Nhịp tim < 60, SpO2 từ 80–100
hr_case1 = np.random.randint(30, 60, size=n_each)
spo2_case1 = np.random.uniform(80, 100, size=n_each)

# Trường hợp 2: Nhịp tim > 100, SpO2 từ 80–100
hr_case2 = np.random.randint(101, 160, size=n_each)
spo2_case2 = np.random.uniform(80, 100, size=n_each)

# Trường hợp 3: SpO2 < 95, Nhịp tim từ 30–160
spo2_case3 = np.random.uniform(80, 95, size=n_each)
hr_case3 = np.random.randint(30, 160, size=n_each)

# Tạo DataFrame cho từng trường hợp
df_case1 = pd.DataFrame({'Heart Rate': hr_case1, 'Oxygen Saturation': spo2_case1, 'Label': [1] * n_each})
df_case2 = pd.DataFrame({'Heart Rate': hr_case2, 'Oxygen Saturation': spo2_case2, 'Label': [1] * n_each})
df_case3 = pd.DataFrame({'Heart Rate': hr_case3, 'Oxygen Saturation': spo2_case3, 'Label': [1] * n_each})

# Gộp lại thành 10.000 mẫu bất thường
df_abnormal = pd.concat([df_case1, df_case2, df_case3], ignore_index=True)

# Loại bỏ các dòng trùng lặp hoàn toàn (nếu có)
df_abnormal_unique = df_abnormal.drop_duplicates()

# (Tùy chọn) Nếu bạn cần đúng 10.000 mẫu sau khi loại trùng, bạn có thể sinh thêm để bù
while len(df_abnormal_unique) < 10000:
    needed = 10000 - len(df_abnormal_unique)
    
    # Tạo thêm từ các trường hợp bất thường (ví dụ dùng lại case 3)
    extra_hr = np.random.randint(30, 160, size=needed)
    extra_spo2 = np.random.uniform(80, 95, size=needed)
    df_extra = pd.DataFrame({'Heart Rate': extra_hr, 'Oxygen Saturation': extra_spo2, 'Label': [1] * needed})
    
    # Gộp lại và loại trùng
    df_abnormal_unique = pd.concat([df_abnormal_unique, df_extra], ignore_index=True).drop_duplicates()

# Reset index sau khi xử lý
df_abnormal_unique = df_abnormal_unique.reset_index(drop=True)

# Lưu ra file
df_abnormal_unique.to_csv("Iot/heart_rate_oxygen_label_1.csv", index=False)
