# Define the new timestamps
from datetime import datetime

time1 = datetime.strptime("2024-06-13 18:08:15", "%Y-%m-%d %H:%M:%S")
time2 = datetime.strptime("2024-06-16 06:34:13", "%Y-%m-%d %H:%M:%S")

# Calculate the difference in seconds
difference_in_seconds = int((time2 - time1).total_seconds())
print(difference_in_seconds)