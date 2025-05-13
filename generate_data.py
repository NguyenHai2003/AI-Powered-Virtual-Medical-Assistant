import pandas as pd

# Đọc dữ liệu từ hai file nhãn 0 và nhãn 1
df_label_0 = pd.read_csv("Iot/heart_rate_oxygen_label_0.csv")
df_label_1 = pd.read_csv("Iot/heart_rate_oxygen_label_1.csv")

# Gộp dữ liệu
df = pd.concat([df_label_0, df_label_1], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle tổng

# Chia tỷ lệ
train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
train_data, val_data, test_data = [], [], []

# Chia theo từng nhãn để cân bằng
for label in df["Label"].unique():
    label_df = df[df["Label"] == label]
    
    train_split = label_df.sample(frac=train_ratio, random_state=42)
    temp_split = label_df.drop(train_split.index)
    val_split = temp_split.sample(frac=val_ratio / (val_ratio + test_ratio), random_state=42)
    test_split = temp_split.drop(val_split.index)
    
    train_data.append(train_split)
    val_data.append(val_split)
    test_data.append(test_split)

# Gộp và shuffle từng tập
train_df = pd.concat(train_data).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat(val_data).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat(test_data).sample(frac=1, random_state=42).reset_index(drop=True)

# Lưu ra file
train_df.to_csv("Iot/heart_spo2_train.csv", index=False)
val_df.to_csv("Iot/heart_spo2_validation.csv", index=False)
test_df.to_csv("Iot/heart_spo2_test.csv", index=False)

# Phân bố nhãn
print("Phân bố nhãn trong tập train:")
print(train_df["Label"].value_counts())

print("\nPhân bố nhãn trong tập validation:")
print(val_df["Label"].value_counts())

print("\nPhân bố nhãn trong tập test:")
print(test_df["Label"].value_counts())

# Kiểm tra trùng dữ liệu giữa các tập
intersection_val_train = pd.merge(val_df, train_df, how='inner')
intersection_test_train = pd.merge(test_df, train_df, how='inner')
intersection_test_val = pd.merge(test_df, val_df, how='inner')

print(f"\nSố dòng trùng giữa validation và train: {len(intersection_val_train)}")
print(f"Số dòng trùng giữa test và train: {len(intersection_test_train)}")
print(f"Số dòng trùng giữa test và validation: {len(intersection_test_val)}")

print("\n✅ Các file CSV đã được tạo:")
print("- Iot/heart_spo2_train.csv")
print("- Iot/heart_spo2_validation.csv")
print("- Iot/heart_spo2_test.csv")