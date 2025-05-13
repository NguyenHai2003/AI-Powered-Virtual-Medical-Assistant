#include <Wire.h>
#include <arduinoFFT.h>
#include "MAX30105.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "disease_prediction_model.h"

MAX30105 particleSensor;

const unsigned long READING_DURATION = 17000;
const unsigned long WAITING_DURATION = 2000;
const unsigned long SAMPLES_PER_SECOND = 100;
const unsigned long SAMPLE_INTERVAL = 1000 / SAMPLES_PER_SECOND;
const int SAMPLES_PER_CALCULATION = 128;
const int TOTAL_SAMPLES = SAMPLES_PER_SECOND * ((READING_DURATION - WAITING_DURATION) / 1000);

uint32_t irBuffer[TOTAL_SAMPLES];
uint32_t redBuffer[TOTAL_SAMPLES];
int bufferIndex = 0;

int heartRates[15];
int spo2Values[15];
int secondIndex = 0;

unsigned long lastSampleTime = 0;
bool isReading = false;
unsigned long readingStartTime = 0;

constexpr int kTensorArenaSize = 64 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

#define SAMPLES 128
#define SAMPLING_FREQ 100

double vReal[SAMPLES];
double vImag[SAMPLES];

ArduinoFFT<double> FFT = ArduinoFFT<double>();

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("Không tìm thấy cảm biến MAX30102!");
    while (1);
  }

  particleSensor.setup(50, 1, 2, 500, 69, 4096);

  tflModel = tflite::GetModel(disease_prediction_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Phiên bản mô hình không khớp!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensor_arena, kTensorArenaSize, &tflErrorReporter);
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  unsigned long currentTime = millis();

  if (particleSensor.getIR() < 50000) {
    if (isReading && (currentTime - lastSampleTime >= 3000)) {
      isReading = false;
      bufferIndex = 0;
      secondIndex = 0;
      Serial.println("Ngắt dữ liệu quá lâu! Hủy phiên đo.");
      return;
    }
    return;
  }

  if (!isReading && particleSensor.getIR() > 50000) {
    isReading = true;
    readingStartTime = currentTime;
    lastSampleTime = currentTime;
    bufferIndex = 0;
    secondIndex = 0;
    Serial.println("Bắt đầu đọc dữ liệu sau 2s...");
  }

  if (isReading && currentTime - readingStartTime >= WAITING_DURATION) {
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
      if (bufferIndex < TOTAL_SAMPLES) {
        irBuffer[bufferIndex] = particleSensor.getIR();
        redBuffer[bufferIndex] = particleSensor.getRed();
        bufferIndex++;
        lastSampleTime = currentTime;
      }

      if (bufferIndex > 0 && bufferIndex % SAMPLES_PER_CALCULATION == 0) {
        int startIdx = bufferIndex - SAMPLES_PER_CALCULATION;
        heartRates[secondIndex] = calculateHeartRateFFT(irBuffer + startIdx, SAMPLES_PER_CALCULATION);
        spo2Values[secondIndex] = calculateSpO2(redBuffer + startIdx, irBuffer + startIdx, SAMPLES_PER_CALCULATION);

        Serial.printf("Giây %d - Nhịp tim: %d BPM, SpO2: %d%%\n", secondIndex + 1, heartRates[secondIndex], spo2Values[secondIndex]);
        secondIndex++;
      }

      if (currentTime - readingStartTime >= READING_DURATION) {
        float avgHeartRate = 0;
        float avgSpO2 = 0;
        int validCount = 0;

        for (int i = 0; i < 10; i++) {
          if (heartRates[i] > 0 && spo2Values[i] > 0) {
            avgHeartRate += heartRates[i];
            avgSpO2 += spo2Values[i];
            validCount++;
          }
        }

        if (validCount > 0) {
          avgHeartRate /= validCount;
          avgSpO2 /= validCount;
          Serial.printf("\nKết quả trung bình:\nNhịp tim: %.1f BPM\nSpO2: %.1f%%\n", avgHeartRate, avgSpO2);
          predictDisease(avgHeartRate, avgSpO2);
        } else {
          Serial.println("Không có đủ dữ liệu hợp lệ!");
        }

        isReading = false;
      }
    }
  }

  particleSensor.check();
}

int calculateHeartRateFFT(uint32_t* irData, int size) {
  if (size < SAMPLES) return -1;

  double mean = 0;
  for (int i = 0; i < SAMPLES; i++) {
    mean += irData[i];
  }
  mean /= SAMPLES;

  for (int i = 0; i < SAMPLES; i++) {
    vReal[i] = (double)irData[i] - mean;
    vImag[i] = 0.0;
  }

  FFT.windowing(vReal, SAMPLES, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
  FFT.compute(vReal, vImag, SAMPLES, FFT_FORWARD);
  FFT.complexToMagnitude(vReal, vImag, SAMPLES);

  double peakFreq = 0;
  double peakValue = 0;
  int lowBound = (int)(0.8 * SAMPLES / SAMPLING_FREQ);
  int highBound = (int)(3.3 * SAMPLES / SAMPLING_FREQ);

  for (int i = lowBound; i <= highBound; i++) {
    if (vReal[i] > peakValue) {
      peakValue = vReal[i];
      peakFreq = (double)i * SAMPLING_FREQ / SAMPLES;
    }
  }

  int heartRate = (int)(peakFreq * 60.0);
  return heartRate;
}

int calculateSpO2(uint32_t* redBuffer, uint32_t* irBuffer, int size) {
  if (size == 0) return -1;

  float meanRed = 0, meanIR = 0;
  for (int i = 0; i < size; i++) {
    meanRed += redBuffer[i];
    meanIR += irBuffer[i];
  }
  meanRed /= size;
  meanIR /= size;

  float stdRed = 0, stdIR = 0;
  for (int i = 0; i < size; i++) {
    stdRed += pow(redBuffer[i] - meanRed, 2);
    stdIR += pow(irBuffer[i] - meanIR, 2);
  }
  stdRed = sqrt(stdRed / size);
  stdIR = sqrt(stdIR / size);

  if (meanRed == 0 || meanIR == 0 || stdIR == 0) return -1;

  float ratio = (stdRed / meanRed) / (stdIR / meanIR);
  float SpO2 = 104 - 17 * ratio;

  if (SpO2 > 100) SpO2 = 100;
  if (SpO2 < 0) SpO2 = 0;

  return (int)SpO2;
}

void predictDisease(int heartRate, float spo2) {
  float scaled_hr = (heartRate - 77.5) / 15.13;
  float scaled_spo2 = (spo2 - 91.875) / 6.0124;

  tflInputTensor->data.f[0] = scaled_hr;
  tflInputTensor->data.f[1] = scaled_spo2;

  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Lỗi khi chạy mô hình!");
    return;
  }

  float* output = tflOutputTensor->data.f;
  const char* labels[] = {
    "Bình thường", "Bất thường", "Bệnh A", "Bệnh B"
  };

  int predictedIndex = 0;
  float maxVal = output[0];
  for (int i = 1; i < tflOutputTensor->dims->data[0]; i++) {
    if (output[i] > maxVal) {
      maxVal = output[i];
      predictedIndex = i;
    }
  }

  Serial.printf("\nInput: [%.1f %.1f], Scaled: [%.6f %.6f]\n",
    heartRate, spo2, scaled_hr, scaled_spo2);
  Serial.printf("Dự đoán: %s\n", labels[predictedIndex]);
  Serial.println("Xác suất cho từng bệnh:");
  for (int i = 0; i < tflOutputTensor->dims->data[0]; i++) {
    Serial.printf("%s: %.2f%%\n", labels[i], output[i] * 100);
  }
  Serial.println("----------------");
}
