import numpy as np
import pandas as pd

np.random.seed(24)

days = 7
hours = days * 24

date_range = pd.date_range(start='2024-01-01', periods=hours, freq='h')

def simulate_cpu_usage(hour):
    if 9 <= hour < 14:
        return np.random.randint(80, 90)
    elif 8 <= hour < 9 or 14 <= hour < 16:
        return np.random.randint(30, 40)
    else:
        return np.random.randint(20, 30)

def simulate_disk_usage(hour):
    if 9 <= hour < 14:
        return np.random.randint(70, 85)
    elif 8 <= hour < 9 or 14 <= hour < 16:
        return np.random.randint(40, 55)
    else:
        return np.random.randint(10, 30)

cpu_usage = [simulate_cpu_usage(dt.hour) for dt in date_range]
disk_usage = [simulate_disk_usage(dt.hour) for dt in date_range]

cpu_data = pd.DataFrame({'DateTime': date_range, 'CPU_Usage': cpu_usage, 'Disk_Usage': disk_usage})

cpu_data.to_csv('data/fabricated_data.csv', index=False)
print(f"Done simulating CPU and Disk usage! \nFabricated {len(cpu_data)} rows.")
