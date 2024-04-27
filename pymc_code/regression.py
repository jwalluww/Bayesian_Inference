import numpy as np
import pymc as pm

# Define the number of data points
num_data_points = 100

# Generate random values for the features
student_gpa = np.random.uniform(low=0.0, high=4.0, size=num_data_points)
past_donations = np.random.uniform(low=0.0, high=100.0, size=num_data_points)
upper_classman = np.random.choice([0, 1], size=num_data_points)

# Generate the target variable (donation rate)
donation_rate = 0.5 + 0.2 * student_gpa + 0.1 * past_donations + 0.3 * upper_classman + np.random.normal(loc=0.0, scale=0.1, size=num_data_points)

# Print the first few rows of the dataset
print("student_gpa, past_donations, upper_classman, donation_rate")
for i in range(num_data_points):
    print(f"{student_gpa[i]:.2f}, {past_donations[i]:.2f}, {upper_classman[i]}, {donation_rate[i]:.2f}")

# Create a PyMC3 model
with pm.Model() as model:
    # Define priors for the regression coefficients
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    beta3 = pm.Normal('beta3', mu=0, sigma=10)

    # Define the linear regression model
    mu = beta0 + beta1 * student_gpa + beta2 * past_donations + beta3 * upper_classman

    # Define the likelihood (normal distribution) for the target variable
    donation_rate_obs = pm.Normal('donation_rate_obs', mu=mu, sigma=0.1, observed=donation_rate)

    # Perform MCMC sampling
    trace = pm.sample(1000, tune=1000)

# Print the summary statistics of the posterior distribution
print(pm.summary(trace))