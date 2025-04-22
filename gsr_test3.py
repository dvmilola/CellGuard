import smbus
import time

# Initialize I2C bus
bus = smbus.SMBus(1)
address = 0x48  # I2C address for PCF8591

sensor_current = 0.001  # 1 mA assumed sensor current

def read_gsr():
    try:
        bus.write_byte(address, 0x40)   # Channel 0
        bus.read_byte(address)         # Dummy read
        time.sleep(0.01)               # Allow ADC to settle
        adc_value = bus.read_byte(address)
    except IOError as e:
        print("I2C read error:", e)
        return 0, 0, 0

    voltage = (adc_value / 255.0) * 3.3
    conductance = (sensor_current / voltage) * 1e6 if voltage > 0 else 0

    return adc_value, voltage, conductance

def get_average_gsr(duration=10):
    total_adc = 0
    total_voltage = 0
    total_conductance = 0
    readings = 0

    print(f"[INFO] Collecting GSR data for {duration} seconds...")
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

    avg_adc = total_adc / readings
    avg_voltage = total_voltage / readings
    avg_conductance = total_conductance / readings

    return avg_adc, avg_voltage, avg_conductance

# Run once and terminate
if __name__ == "__main__":
    avg_adc, avg_volt, avg_cond = get_average_gsr(duration=10)

    print("\n--- GSR Results ---")
    print(f"Average ADC Value   : {avg_adc:.2f}")
    print(f"Average Voltage     : {avg_volt:.2f} V")
    print(f"Average Conductance : {avg_cond:.2f} µS")
    print("---------------------")
