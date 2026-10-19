"""RainCast IoT node – MicroPython (ESP32 / Raspberry Pi Pico W).

Copy main.py and a BME280 driver (e.g. `bme280_float.py`) to the board.
Same API contract as the Arduino sketch: POST readings to /api/iot.
"""
import time

import network
import urequests
from machine import ADC, I2C, Pin

import bme280_float as bme280

WIFI_SSID, WIFI_PASS = "YOUR_WIFI", "YOUR_PASSWORD"
API_URL = "https://your-app.vercel.app/api/iot"
DEVICE_ID = "esp32-mpy-01"
LAT, LON = 19.0728, 72.8826
INTERVAL_S = 600

i2c = I2C(0, scl=Pin(22), sda=Pin(21))
sensor = bme280.BME280(i2c=i2c)
rain_adc = ADC(Pin(34))
rain_adc.atten(ADC.ATTN_11DB)
led = {"red": Pin(25, Pin.OUT), "green": Pin(26, Pin.OUT), "blue": Pin(27, Pin.OUT)}
history = []


def wifi():
    sta = network.WLAN(network.STA_IF)
    sta.active(True)
    if not sta.isconnected():
        sta.connect(WIFI_SSID, WIFI_PASS)
        for _ in range(40):
            if sta.isconnected():
                break
            time.sleep(0.5)
    return sta.isconnected()


def show(colour):
    led["red"].value(colour in ("red", "amber"))
    led["green"].value(colour in ("green", "amber"))
    led["blue"].value(0)


while True:
    t, p, h = sensor.read_compensated_data()
    p = p / 100
    history.append(p)
    history[:] = history[-18:]
    payload = {
        "device_id": DEVICE_ID, "temperature": round(t, 2), "humidity": round(h, 1),
        "pressure": round(p, 1), "pressure_3h_ago": round(history[0], 1),
        "rain_analog": sum(rain_adc.read() for _ in range(8)) // 8, "adc_max": 4095,
        "lat": LAT, "lon": LON,
    }
    if wifi():
        try:
            r = urequests.post(API_URL, json=payload)
            res = r.json()
            r.close()
            print(res["rain_probability"], res["rain_type"])
            show(res["led"])
        except Exception as exc:  # network hiccup – try again next cycle
            print("error", exc)
    time.sleep(INTERVAL_S)
