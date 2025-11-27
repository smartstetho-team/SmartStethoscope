#include <Arduino.h>

const int micPin        = 1;     // MAX4466 output -> ADC pin on ESP32-S3
const int sampleRate    = 8000;  // Hz
const int recordSeconds = 10;    // seconds
const int numSamples    = sampleRate * recordSeconds;

void setup() {
  Serial.begin(115200);
  delay(2000);                   // let USB come up

  analogReadResolution(12);      // 0..4095

  Serial.println("READY");       // so Python can read one line if it wants
  Serial.print("Will record ");
  Serial.print(recordSeconds);
  Serial.println(" s when I get 'r'");
}

void recordAndSend() {
  Serial.println("Recording...");

  for (int i = 0; i < numSamples; i++) {
    int sample = analogRead(micPin);   // 0..4095

    // Send as unsigned 16-bit big-endian: high byte then low byte
    Serial.write((sample >> 8) & 0xFF);
    Serial.write(sample & 0xFF);

    delayMicroseconds(1000000 / sampleRate);
  }

  Serial.println("\nDONE");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r') {
      recordAndSend();
      // after sending one clip, you can either:
      // - stay idle and wait for another 'r'
      // - or stop. Here we wait for another 'r'.
      Serial.println("READY");
    }
  }
}
