/*
 * RainCast IoT node – ESP32 + BME280 + FC-37 rain sensor
 * -------------------------------------------------------
 * Reads temperature, humidity, pressure and rain-plate wetness every 10 min,
 * POSTs them to the RainCast API (/api/iot) on Vercel and drives an RGB LED:
 *   green < 40 %  ·  amber 40-69 %  ·  red >= 70 %  (+ buzzer on heavy rain / storm)
 *
 * Libraries (Arduino Library Manager):
 *   Adafruit BME280, Adafruit Unified Sensor, ArduinoJson (v7)
 *
 * Wiring:
 *   BME280  SDA -> GPIO21, SCL -> GPIO22, VCC 3V3, GND
 *   FC-37   AO  -> GPIO34 (ADC1), VCC 3V3, GND
 *   RGB LED R/G/B -> GPIO25/26/27 (220 Ω resistors), common cathode -> GND
 *   Buzzer  -> GPIO14
 */
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <ArduinoJson.h>

// ---------- configure ----------
const char* WIFI_SSID   = "YOUR_WIFI";
const char* WIFI_PASS   = "YOUR_PASSWORD";
const char* API_URL     = "https://your-app.vercel.app/api/iot";
const char* DEVICE_ID   = "esp32-roof-01";
const char* DEVICE_KEY  = "";          // matches IOT_API_KEY env var on Vercel (optional)
const float LATITUDE    = 19.0728;     // set to your location to blend with live data
const float LONGITUDE   = 72.8826;
const uint32_t INTERVAL_MS = 10UL * 60UL * 1000UL;
// --------------------------------

#define PIN_RAIN 34
#define PIN_R 25
#define PIN_G 26
#define PIN_B 27
#define PIN_BUZZ 14

Adafruit_BME280 bme;
float pressureHistory[18];   // 18 x 10 min = 3 h of pressure readings
int histIdx = 0, histCount = 0;

void setLed(const String& c) {
  digitalWrite(PIN_R, c == "red" || c == "amber");
  digitalWrite(PIN_G, c == "green" || c == "amber");
  digitalWrite(PIN_B, LOW);
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) { delay(500); Serial.print("."); }
  Serial.println(WiFi.status() == WL_CONNECTED ? " connected" : " failed");
}

int readRainAnalog() {
  long sum = 0;
  for (int i = 0; i < 16; i++) { sum += analogRead(PIN_RAIN); delay(5); }
  return sum / 16;   // 0 (soaked) .. 4095 (dry)
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_R, OUTPUT); pinMode(PIN_G, OUTPUT); pinMode(PIN_B, OUTPUT); pinMode(PIN_BUZZ, OUTPUT);
  digitalWrite(PIN_B, HIGH);   // blue = booting
  if (!bme.begin(0x76) && !bme.begin(0x77)) Serial.println("BME280 not found – check wiring");
  analogReadResolution(12);
  connectWiFi();
}

void loop() {
  float t = bme.readTemperature();
  float h = bme.readHumidity();
  float p = bme.readPressure() / 100.0F;   // Pa -> hPa
  int rain = readRainAnalog();

  pressureHistory[histIdx] = p;
  histIdx = (histIdx + 1) % 18;
  if (histCount < 18) histCount++;
  float p3h = histCount == 18 ? pressureHistory[histIdx] : pressureHistory[0];

  JsonDocument doc;
  doc["device_id"] = DEVICE_ID;
  doc["temperature"] = t;
  doc["humidity"] = h;
  doc["pressure"] = p;
  doc["pressure_3h_ago"] = p3h;
  doc["rain_analog"] = rain;
  doc["adc_max"] = 4095;
  doc["lat"] = LATITUDE;
  doc["lon"] = LONGITUDE;
  String body;
  serializeJson(doc, body);
  Serial.println("-> " + body);

  connectWiFi();
  if (WiFi.status() == WL_CONNECTED) {
    WiFiClientSecure client;
    client.setInsecure();               // demo only – pin the Vercel root CA in production
    HTTPClient http;
    http.begin(client, API_URL);
    http.addHeader("Content-Type", "application/json");
    if (strlen(DEVICE_KEY)) http.addHeader("X-Device-Key", DEVICE_KEY);
    int code = http.POST(body);
    if (code == 200) {
      JsonDocument res;
      deserializeJson(res, http.getString());
      int prob = res["rain_probability"];
      String type = res["rain_type"].as<String>();
      String led = res["led"].as<String>();
      Serial.printf("<- %d%% %s (led %s)\n", prob, type.c_str(), led.c_str());
      setLed(led);
      if (res["alert"]) { for (int i = 0; i < 3; i++) { tone(PIN_BUZZ, 2200, 200); delay(350); } }
    } else {
      Serial.printf("HTTP error %d\n", code);
      digitalWrite(PIN_B, HIGH);
    }
    http.end();
  }
  delay(INTERVAL_MS);
}
