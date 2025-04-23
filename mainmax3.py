import time
import math
import traceback
from periphery import I2C

# MAX30102 I2C address
MAX30102_ADDR = 0x57

# Register addresses
REG_INT_STATUS = 0x00
REG_INT_ENABLE = 0x02
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_DATA = 0x07
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA = 0x0C
REG_LED2_PA = 0x0D
REG_PART_ID = 0xFF

# I2C initialization
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
        if part_id == 0x15:
            print(f"MAX30102 sensor detected (ID: 0x{part_id:02X})")
            return True
        else:
            print(f"Unknown device ID: 0x{part_id:02X}")
            return False
    except Exception as e:
        print(f"Error detecting sensor: {e}")
        traceback.print_exc()
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
    print("MAX30102 initialized.")

def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std_dev(data):
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def find_peaks(data, min_distance=25, threshold_factor=0.7):
    peaks = []
    mean_value = calculate_mean(data)
    std_dev = calculate_std_dev(data)
    threshold = mean_value + threshold_factor * std_dev
    i = min_distance
    while i < len(data) - min_distance:
        if data[i] > threshold:
            if data[i] >= max(data[i-min_distance:i+min_distance+1]):
                peaks.append(i)
                i += min_distance
            else:
                i += 1
        else:
            i += 1
    return peaks

def calculate_heart_rate(ir_values, sampling_rate=100):
    if len(ir_values) < sampling_rate * 3:
        return 0

    mean_ir = calculate_mean(ir_values)
    normalized_ir = [val - mean_ir for val in ir_values]

    window_size = 5
    smoothed_ir = []
    for i in range(len(normalized_ir)):
        start = max(0, i - window_size // 2)
        end = min(len(normalized_ir), i + window_size // 2 + 1)
        smoothed_ir.append(sum(normalized_ir[start:end]) / (end - start))

    peaks = find_peaks(smoothed_ir, min_distance=int(sampling_rate * 0.5))

    if len(peaks) < 2:
        return 0

    intervals = []
    for i in range(1, len(peaks)):
        interval_seconds = (peaks[i] - peaks[i - 1]) / sampling_rate
        if 0.3 <= interval_seconds <= 2.0:
            intervals.append(interval_seconds)

    if not intervals:
        return 0

    avg_interval = sum(intervals) / len(intervals)
    heart_rate = 60 / avg_interval

    return round(heart_rate, 1) if 40 <= heart_rate <= 220 else 0

def calculate_spo2(red_values, ir_values):
    if len(red_values) < 100 or calculate_mean(ir_values) < 1000:
        return 0

    red_mean = calculate_mean(red_values)
    ir_mean = calculate_mean(ir_values)
    red_ac = [abs(val - red_mean) for val in red_values]
    ir_ac = [abs(val - ir_mean) for val in ir_values]

    red_rms = math.sqrt(calculate_mean([val ** 2 for val in red_ac]))
    ir_rms = math.sqrt(calculate_mean([val ** 2 for val in ir_ac]))

    r = (red_rms / red_mean) / (ir_rms / ir_mean)
    spo2 = 110 - 25 * r

    return round(spo2, 1) if 70 <= spo2 <= 100 else 0

def read_sensor_data(duration=10, sampling_rate=100):
    red_readings = []
    ir_readings = []

    print("Please place your finger on the sensor...")
    print("Reading will start in 3 seconds...")
    time.sleep(3)

    print("Reading sensor data...")
    reset_sensor()
    start_time = time.time()
    sample_count = 0

    while time.time() - start_time < duration:
        sample_start = time.time()
        try:
            red, ir = read_fifo()
            if ir > 1000:
                red_readings.append(red)
                ir_readings.append(ir)
                sample_count += 1
                if sample_count % 25 == 0:
                    print(f"Samples: {sample_count}, RED: {red}, IR: {ir}")
            else:
                print("Weak signal - adjust finger placement")
        except Exception as e:
            print(f"Read error: {e}")
            traceback.print_exc()

        elapsed = time.time() - sample_start
        if elapsed < 1 / sampling_rate:
            time.sleep(1 / sampling_rate - elapsed)

    print(f"Collected {len(ir_readings)} samples.")
    return red_readings, ir_readings

def main():
    print("MAX30102 Heart Rate and SpO₂ Monitor")
    print("------------------------------------")

    if not check_sensor():
        print("Sensor not found. Check wiring.")
        return

    setup_sensor()
    red, ir = read_sensor_data(duration=10)

    if len(ir) > 100:
        heart_rate = calculate_heart_rate(ir)
        spo2 = calculate_spo2(red, ir)

        if heart_rate > 0:
            print(f"\nHeart Rate: {heart_rate} BPM")
        else:
            print("\nCould not compute heart rate.")

        if spo2 > 0:
            print(f"SpO₂: {spo2}%")
        else:
            print("Could not compute SpO₂.")
    else:
        print("Insufficient data. Try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
