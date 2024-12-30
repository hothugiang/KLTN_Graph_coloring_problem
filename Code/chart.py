import matplotlib.pyplot as plt

# Example data: average values and standard deviations
x = [1, 2, 3, 4, 5]  # x-axis values (could represent time, categories, etc.)
y = [10, 20, 25, 30, 35]  # average values (y-axis)
std_dev = [2, 3, 2, 1, 4]  # standard deviations (error bars)

# Create the plot with error bars
plt.errorbar(x, y, yerr=std_dev, fmt='-o', capsize=5, linestyle='-', color='b', ecolor='r', elinewidth=2, capthick=2)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Average Values')
plt.title('Line Chart with Error Bars')

# Show the plot
plt.show()