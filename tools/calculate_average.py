import re

# Original string containing \textbf{} tags and other numbers
data = "77.79 & \textbf{96.89} & 92.98 & 90.93 & 65.41 & \textbf{98.67} & \textbf{97.81} & \textbf{97.51} & 94.27 & \textbf{96.39} & 94.79 & 91.72 & \textbf{100.00} & \textbf{96.82} & \textbf{100.00}"

# Use regex to remove '\textbf{}' and '&' symbols, and extract all numbers
cleaned_data = re.findall(r"\d+\.\d+", data)

# Convert the list of strings to a list of floats
numbers = list(map(float, cleaned_data))

# Calculate the mean value of the numbers
mean_value = sum(numbers) / len(numbers)

# Print the length of the numbers list
print(f"Length of numbers list: {len(numbers)}")

# Print the mean value
print(f"The mean value is: {mean_value}")
