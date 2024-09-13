# Define the new timestamps
from datetime import datetime

time1 = datetime.strptime("2024-08-07 11:07:07", "%Y-%m-%d %H:%M:%S")
time2 = datetime.strptime("2024-08-07 11:07:36", "%Y-%m-%d %H:%M:%S")

# Calculate the difference in seconds
difference_in_seconds = int((time2 - time1).total_seconds())
print(difference_in_seconds)