# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:31:59 2023

@author: lenovo
"""

##                Cost estimation model with average value
def calculate_cost(capex, fom, vom, heatr, ngp, eff, er, ep, discount, year):
    discounted_value =  1 / (1 + discount) ** (year-1)
    return discounted_value



capex = 467  # Capital costs
fom = 15.04     # Fixed OPEX
vom = 4.61    # Variable OPEX
heatr = 3.4  # reboiler duty
ngp = 6.93       # Natural gas price
er = 156       #Electricity requirement
ep = 0.1035    #Electricity price 
discount = 0.05  # Discount rate
eff = 0.8        # Boiler efficiency
ts = 10
discount_factor = 0

for year in range(1, 26):
    discounted_value = calculate_cost(capex, fom, vom, heatr, ngp, eff, er, ep, discount, year)
    discount_factor += discounted_value
    print(f"Discounted value for year {year}: {discounted_value}")

total_cost = capex / discount_factor + fom + vom + heatr * ngp / eff + er*ep + ts
print(f"Total Cost: {total_cost}")




##                Define PDF for inputs and random sampling
##Natural gas price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Load data from Excel
data_frame = pd.read_excel('distributionCC.xlsx', sheet_name='Sheet1')
ngpd = data_frame['ngp'].values
hist_ngp, hist_bins_ngp = np.histogram(ngpd, bins=30, density=True)
bin_wid_ngp = np.diff(hist_bins_ngp)
# Approximate the PDF from the histogram
pdf_ngp = hist_ngp / np.sum(hist_ngp * bin_wid_ngp)
# Fit a KDE curve to the data
kde_ngp = gaussian_kde(ngpd)
# Define the range of values for plotting
x_range_ngp = np.linspace(min(ngpd), max(ngpd), 10000)
# Plot the histogram, the KDE curve, and the PDF
plt.hist(ngpd, bins=30, density=True, alpha=0.6, color='b', label='Data Histogram')
plt.plot(x_range_ngp, kde_ngp(x_range_ngp), 'r', label='KDE PDF')
plt.plot(hist_bins_ngp[:-1], pdf_ngp, 'g', label='Approximate PDF')
plt.xlabel('Natural Gas Price (£/GJ)')
plt.ylabel('Probability Density')
plt.title('Natural Gas Price PDF and KDE')
plt.legend()
plt.show()
# Generate random samples based on the KDE
num_samples = 10000
random_ngp = kde_ngp.resample(num_samples)[0]
# Visualization of random sampling data
plt.hist(random_ngp, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Natural Gas Price (£/GJ)')
plt.ylabel('Probability Density')
plt.title('Natural Gas Price Input')
plt.grid(True)
plt.show()



##Electricity price
# Load data from Excel
data_frame = pd.read_excel('distributionCC.xlsx', sheet_name='Sheet1')
epd = data_frame['ep'].values
# Create a histogram
hist_ep, hist_bins_ep = np.histogram(epd, bins=30, density=True)
bin_wid_ep = np.diff(hist_bins_ep)
# Approximate the PDF from the histogram
pdf_ep = hist_ep / np.sum(hist_ep * bin_wid_ep)
# Fit a KDE curve to the data
kde_ep = gaussian_kde(epd)
# Define the range of values for plotting
x_range_ep = np.linspace(min(epd), max(epd), 10000)
# Plot the histogram, the KDE curve, and the PDF
plt.hist(epd, bins=20, density=True, alpha=0.6, color='b', label='Data Histogram')
plt.plot(x_range_ep, kde_ep(x_range_ep), 'r', label='KDE PDF')
plt.plot(hist_bins_ep[:-1], pdf_ep, 'g', label='Approximate PDF')
plt.xlabel('Electricity Price (£/kWh)')
plt.ylabel('Probability Density')
plt.title('Electricity Price PDF and KDE')
plt.legend()
plt.show()
# Generate random samples based on the KDE
num_samples = 10000
random_ep = kde_ep.resample(num_samples)[0]
# Visualization of random sampling data
plt.hist(random_ep, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Electricity Price (£/kWh)')
plt.ylabel('Probability Density')
plt.title('Electricity Price Input')
plt.grid(True)
plt.show()



##CAPEX
random_capex = np.random.uniform(109.6,769.46,10000)
# Visualisation
plt.hist(random_capex, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('CAPEX (M£)')
plt.ylabel('Probability Density')
plt.title('CAPEX Input')
plt.grid(True)
plt.show()



##Variable OPEX
random_vom = np.random.uniform(1.6,7.25,10000)
# Visualisation
plt.hist(random_vom, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Variable OPEX (M£/y)')
plt.ylabel('Probability Density')
plt.title('Variable OPEX Cost Input')
plt.grid(True)
plt.show()



##Boiler efficiency
random_eff = np.random.uniform(0.7,0.9,10000)
# Visualisation
plt.hist(random_eff, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Reboiler Efficiency')
plt.ylabel('Probability Density')
plt.title('Reboiler Efficiency Input')
plt.grid(True)
plt.show()



##Discount rate
random_discount = np.random.uniform(0.03,0.08,10000)
# Visualisation
plt.hist(random_discount, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Discount Rate')
plt.ylabel('Probability Density')
plt.title('CapEx Input')
plt.grid(True)
plt.show()



##Heat consumption
# Generate 1000 random numbers using rejection sampling for heatr
random_heatr = []
while len(random_heatr) < 10000:
    sample_heatr = np.random.normal(loc=3.4, scale=0.87, size=1)
    if 1.8 <= sample_heatr[0] <= 9.2:  # Access the value from the numpy array
        random_heatr.append(sample_heatr[0])  # Append the value to the list
# Visualisation
plt.hist(random_heatr, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Heat Requirement (GJ/t CO2)')
plt.ylabel('Probability Density')
plt.title('Heat Requirement Input')
plt.grid(True)
plt.show()



##Electricity requirement 
random_er = []
while len(random_er) < 10000:
    sample_er = np.random.normal(loc=156, scale=33, size=1)
    if 120 <= sample_er[0] <= 194:  # Access the value from the numpy array
        random_er.append(sample_er[0])  # Append the value to the list
# Visualisation
plt.hist(random_er, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel('Electricity Requirement (kWh/t CO2)')
plt.ylabel('Probability Density')
plt.title('Electricity Requirement Input')
plt.grid(True)
plt.show()



## Visualisation of all input data
plt.figure(figsize=(20, 15))
# Create subplots for each column
columns = [
    ("CAPEX (M£)", random_capex, 1),
    ("Variable OPEX (M£/y)", random_vom, 2),
    ("Reboiler Duty (PJ/y)", random_heatr, 3),
    ("Natural Gas Price (£/GJ)", random_ngp, 4),
    ("Boiler Efficiency", random_eff, 5),
    ("Electricity Requirement (GWh/y)", random_er, 6),
    ("Electricity Price (£/kWh)", random_ep, 7),
    ("Discount Rate", random_discount, 8),
]

for title, data, position in columns:
    plt.subplot(3, 3, position)
    plt.hist(data, bins=50, color='blue')
    plt.title(title, fontsize=20)
    plt.xlabel("Value", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

# Adjust layout
plt.tight_layout()
# Show the big figure with all subplots
plt.show()




##            Global sensitivity analysis with cost model
import seaborn as sns

def gsa_calculate_cost(random_capex, fom, random_vom, random_heatr, 
                       random_ngp, random_eff, random_er, random_ep, 
                       random_discount, year, ts):
    gsa_discounted_value =  1 / (1 + random_discount) ** (year-1)
    return gsa_discounted_value

gsa_output_costs = []

for i in range(10000):
    gsa_discount_factor = 0
    for year in range(1, 26):
        gsa_discounted_value = gsa_calculate_cost(
            random_capex[i], fom, random_vom[i], random_heatr[i], 
            random_ngp[i], random_eff[i], random_er[i], random_ep[i],
            random_discount[i], year, ts
        )
        gsa_discount_factor += gsa_discounted_value
    
    gsa_total = random_capex[i] / gsa_discount_factor + fom + random_vom[i] + random_heatr[i] * random_ngp[i] / random_eff[i] + random_er[i] * random_ep[i] + ts
    gsa_output_costs.append(gsa_total)
# Visualize the distribution of output costs
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(gsa_output_costs, bins=50, kde=True, color='blue')
plt.title("Distribution of Total Avoidance Costs", fontsize=20)
plt.xlabel("Total Avoidance Cost (£/tCO$_{2}$)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()




##                      Corelation between input and output
# Create scatter plots for each input parameter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

input_params = [random_capex, random_vom, random_heatr, random_ngp, random_eff,
                random_er, random_ep, random_discount]
input_param_names = ['CAPEX (M£)', 'Variable OPEX (M£/y)', 'Reboiler Duty (PJ/y)', 'Natural Gas Price (£/GJ)',
                     'Boiler Efficiency', 'Electricity Requirement (GWh/y)', 'Electricity Price (£/kWh)', 'Discount Rate', ]

plt.figure(figsize=(20, 15))
for i, param in enumerate(input_params):
    plt.subplot(3, 3, i+1)
    param = np.array(param)  # Convert to a NumPy array if it's a list
    plt.scatter(param, gsa_output_costs, alpha=0.5)
    plt.xlabel(input_param_names[i], fontsize=16)
    plt.ylabel("Total Avoidance Cost (£/tCO$_{2}$)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Perform linear regression
    lr = LinearRegression()
    param_reshaped = param.reshape(-1, 1)  # Reshape to fit the linear regression model
    lr.fit(param_reshaped, gsa_output_costs)
    y_pred = lr.predict(param_reshaped)
    # Plot the linear regression line
    plt.plot(param, y_pred, color='Black', linewidth=2)  # Use 'param' for x-values
    # Calculate and display R-squared
    r2 = r2_score(gsa_output_costs, y_pred)
    plt.text(0.95, 0.05, f"R-squared = {r2:.2f}", 
             transform=plt.gca().transAxes, horizontalalignment='right', 
             verticalalignment='bottom', fontsize=20)
plt.tight_layout()
plt.show()

## for significant correlation
input_params = [random_capex, random_heatr, random_ngp, 
                random_ep, random_discount, random_vom]
input_param_names = ['CAPEX (M£)', 'Reboiler Duty (PJ/y)', 'Natural Gas Price (£/GJ)',
                     'Electricity Price (£/kWh)', 'Discount Rate', 'Variable OPEX (M£/y)']

plt.figure(figsize=(20, 10))
for i, param in enumerate(input_params):
    plt.subplot(2, 3, i+1)
    param = np.array(param)  # Convert to a NumPy array if it's a list
    plt.scatter(param, gsa_output_costs, alpha=0.3)
    plt.xlabel(input_param_names[i], fontsize=16)
    plt.ylabel("Total Avoidance Cost (£/tCO$_{2}$)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Perform linear regression
    lr = LinearRegression()
    param_reshaped = param.reshape(-1, 1)  # Reshape to fit the linear regression model
    lr.fit(param_reshaped, gsa_output_costs)
    y_pred = lr.predict(param_reshaped)
    # Plot the linear regression line
    plt.plot(param, y_pred, color='Black', linewidth=2)  # Use 'param' for x-values
    # Calculate and display R-squared
    r2 = r2_score(gsa_output_costs, y_pred)
    plt.text(0.95, 0.05, f"R-squared = {r2:.2f}", 
             transform=plt.gca().transAxes, horizontalalignment='right', 
             verticalalignment='bottom', fontsize=14)
plt.tight_layout()
plt.show()

