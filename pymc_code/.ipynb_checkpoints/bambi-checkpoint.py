import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
import bambi as bmb

from pymc import HalfCauchy, Model, Normal, sample

print(f"Running on PyMC v{pm.__version__}")

# Define the number of data points
num_data_points = 100

# Generate random values for the features
student_gpa = np.random.uniform(low=0.0, high=4.0, size=num_data_points)
past_donations = np.random.uniform(low=0.0, high=100.0, size=num_data_points)
upper_classman = np.random.choice([0, 1], size=num_data_points)

# Generate the target variable (donation rate)
donation_rate = 0.5 + 0.2 * student_gpa + 0.1 * past_donations + 0.3 * upper_classman + np.random.normal(loc=0.0, scale=0.1, size=num_data_points)

# Create a DataFrame
data = {
    'Student_GPA': student_gpa,
    'Past_Donations': past_donations,
    'Upper_Classman': upper_classman,
    'Donation_Rate': donation_rate
}

df = pd.DataFrame(data)

model = bmb.Model("Donation_Rate ~ Past_Donations", df)
idata = model.fit(draws=3000)