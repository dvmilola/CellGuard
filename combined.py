# combined_sensor_reader.py

import time
import math
import traceback
from periphery import I2C
import smbus

### ---------- GSR SENSOR (PCF8591) SETUP ---------- ###
bus = smbus.SMBus(1)
GSR_ADDRESS = 0x48
sensor_current = 0.001  # 1 mA assumed

def read_gsr():
    try:
        bus.write_byte(GSR_ADDRESS, 0x40)
        bus.read_byte(GSR_ADDRESS)
        time.sleep(0.01)
        adc_value = bus.read_byte(GSR_ADDRESS)
    except IOError as e:
        print("GSR I2C read error:", e)
        return 0, 0, 0

    voltage = (adc_value / 255.0) * 3.3
    conductance = (sensor_current / voltage) * 1e6 if voltage > 0 else 0
    return adc_value, voltage, conductance

def get_average_gsr(duration=10):
    total_adc = total_voltage = total_conductance = 0
    readings = 0
    print(f"[GSR] Collecting data for {duration} seconds...")
    start_time = time.time()

    while time.time() - start_time < duration:
        adc, volt, cond = read_gsr()
        total_adc += adc
        total_voltage += volt
        total_conductance += cond
        readings += 1
        time.sleep(0.1)

    if readings == 0:
        return 0, 0, 0

    return total_adc / readings, total_voltage / readings, total_conductance / readings


### ---------- MAX30102 SENSOR SETUP ---------- ###
MAX30102_ADDR = 0x57
REG_FIFO_DATA = 0x07
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
REG_INT_ENABLE = 0x02
REG_PART_ID = 0xFF
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06

i2c = I2C("/dev/i2c-1")

def write_register(reg, value):
    i2c.transfer(MAX30102_ADDR, [I2C.Message([reg, value])])

def read_register(reg):
    messages = [I2C.Message([reg]), I2C.Message(bytearray(1), read=True)]
    i2c.transfer(MAX30102_ADDR, messages)
    return messages[1].data[0]

def check_sensor():
    try:
        part_id = read_register(REG_PART_ID)
        return part_id == 0x15
    except:
        return False

def reset_sensor():
    write_register(REG_FIFO_WR_PTR, 0x00)
    write_register(REG_OVF_COUNTER, 0x00)
    write_register(REG_FIFO_RD_PTR, 0x00)

def read_fifo():
    messages = [I2C.Message([REG_FIFO_DATA]), I2C.Message(bytearray(6), read=True)]
    i2c.transfer(MAX30102_ADDR, messages)
    data = messages[1].data
    red = (data[0] << 16 | data[1] << 8 | data[2]) & 0x3FFFF
    ir = (data[3] << 16 | data[4] << 8 | data[5]) & 0x3FFFF
    return red, ir

def setup_sensor():
    write_register(REG_MODE_CONFIG, 0x40)
    time.sleep(0.1)
    while read_register(REG_MODE_CONFIG) & 0x40:
        time.sleep(0.1)
    reset_sensor()
    write_register(REG_MODE_CONFIG, 0x03)
    write_register(REG_SPO2_CONFIG, 0x27)
    write_register(REG_LED1_PA, 0x24)
    write_register(REG_LED2_PA, 0x24)
    write_register(REG_INT_ENABLE, 0x10)

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std_dev(data):
    mean = calculate_mean(data)
    return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))

def find_peaks(data, min_distance=25, threshold_factor=0.7):
    peaks = []
    mean_value = calculate_mean(data)
    std_dev = calculate_std_dev(data)
    threshold = mean_value + threshold_factor * std_dev
    i = min_distance
    while i < len(data) - min_distance:
        if data[i] > threshold and data[i] >= max(data[i-min_distance:i+min_distance+1]):
            peaks.append(i)
            i += min_distance
        else:
            i += 1
    return peaks

def calculate_heart_rate(ir_values, sampling_rate=100):
    if len(ir_values) < sampling_rate * 3:
        return 0
    mean_ir = calculate_mean(ir_values)
    norm = [val - mean_ir for val in ir_values]
    smoothed = [sum(norm[max(0, i-2):i+3]) / len(norm[max(0, i-2):i+3]) for i in range(len(norm))]
    peaks = find_peaks(smoothed, min_distance=int(sampling_rate * 0.5))
    intervals = [(peaks[i] - peaks[i-1]) / sampling_rate for i in range(1, len(peaks))]
    if not intervals:
        return 0
    return round(60 / (sum(intervals) / len(intervals)), 1)

def calculate_spo2(red, ir):
    if len(red) < 100:
        return 0
    r_mean, ir_mean = calculate_mean(red), calculate_mean(ir)
    r_rms = math.sqrt(sum([(x - r_mean)**2 for x in red]) / len(red))
    ir_rms = math.sqrt(sum([(x - ir_mean)**2 for x in ir]) / len(ir))
    r = (r_rms / r_mean) / (ir_rms / ir_mean)
    spo2 = 110 - 25 * r
    return round(spo2, 1) if 70 <= spo2 <= 100 else 0

def read_sensor_data(duration=10, sampling_rate=100):
    red_readings = []
    ir_readings = []
    print("[MAX30102] Reading in 3 seconds...")
    time.sleep(3)
    reset_sensor()
    start = time.time()
    while time.time() - start < duration:
        try:
            red, ir = read_fifo()
            if ir > 1000:
                red_readings.append(red)
                ir_readings.append(ir)
        except:
            continue
        time.sleep(1 / sampling_rate)
    return red_readings, ir_readings


### ---------- COMBINED MAIN FUNCTION ---------- ###
def main():
    print("Starting Data Collection from GSR and MAX30102...")
    
    # Step 1: Collect GSR Data
    gsr_adc, gsr_volt, gsr_cond = get_average_gsr(duration=10)
    print("\n[GSR RESULTS]")
    print(f"Avg ADC       : {gsr_adc:.2f}")
    print(f"Avg Voltage   : {gsr_volt:.2f} V")
    print(f"Avg Conduct.  : {gsr_cond:.2f} µS")

    # Step 2: Collect MAX30102 Data
    if not check_sensor():
        print("[ERROR] MAX30102 not found.")
        return

    setup_sensor()
    red_vals, ir_vals = read_sensor_data(duration=10)

    if len(ir_vals) >= 100:
        hr = calculate_heart_rate(ir_vals)
        spo2 = calculate_spo2(red_vals, ir_vals)
        print("\n[MAX30102 RESULTS]")
        print(f"Heart Rate    : {hr} BPM")
        print(f"SpO₂          : {spo2}%")
    else:
        print("[ERROR] Not enough data from MAX30102.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
