import serial
import re


def read_heart_data(port="COM3", baud_rate=115200, timeout=2):
    try:
        # Kết nối Serial
        ser = serial.Serial(port, baud_rate, timeout=timeout)
        print("Đang chờ dữ liệu từ thiết bị...")

        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                bpm_match = re.search(r"Nhịp tim: (\d+\.\d+|\d+) BPM", line)
                spo2_match = re.search(r"SpO2: (\d+\.\d+|\d+)%", line)
                prediction_match = re.search(r"Dự đoán: (.+)", line)

                bpm = float(bpm_match.group(1)) if bpm_match else None
                spo2 = float(spo2_match.group(1)) if spo2_match else None
                prediction = prediction_match.group(1) if prediction_match else None

                if bpm is not None and spo2 is not None:
                    return bpm, spo2, prediction  # Trả về kết quả khi có đủ dữ liệu

    except serial.SerialException as e:
        print(f"Lỗi kết nối Serial: {e}")
        return None, None, None
    finally:
        ser.close()



