import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

data_dict = {
    "Year": [1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    "Exchange Rate": [4.76, 4.76, 11.01, 9.99, 9.99, 9.99, 9.99, 9.99, 9.99, 9.99, 9.99, 9.99, 11.85, 13.12, 14.05, 15.93, 16.65, 17.4, 18, 20.54, 21.71, 23.8, 25.08, 28.11, 30.57, 31.64, 36.08, 41.11, 45.05, 51.9, 51.9, 63.5, 60.5, 57.75, 57.8, 59.7, 60.4, 60.83, 81.1, 84.1, 85.75, 88.6, 96.5, 107.2, 103, 105.2, 104.6, 110.01, 139, 163.75, 168.88, 179.16, 225.4],
    "Inflation Rate": [5.35, 4.73, 5.18, 23.07, 26.66, 20.9, 7.16, 10.13, 6.14, 8.27, 11.94, 11.88, 5.9, 6.36, 6.09, 5.61, 3.51, 4.68, 8.84, 7.84, 9.05, 11.79, 9.51, 9.97, 12.37, 12.34, 10.37, 11.38, 6.23, 4.14, 4.37, 3.15, 3.29, 2.91, 7.44, 9.06, 7.92, 7.6, 20.29, 13.65, 12.94, 11.92, 9.68, 7.69, 7.19, 2.53, 3.77, 4.09, 5.08, 10.58, 9.74, 9.5, 19.87],
    "Central Government Debt": [35.01, 36.81, 98.34, 48.11, 42.11, 35.58, 39.75, 40.37, 40.09, 40.77, 36.91, 32.89, 37.45, 37.24, 37.02, 41.08, 47.76, 49.21, 49.87, 51.7, 53.54, 61.85, 59.14, 64.25, 64.83, 57.96, 58.19, 58.46, 59.49, 67.2, 68.39, 72.15, 67.58, 62.68, 56.3, 52.34, 48.4, 47.08, 51.88, 52.82, 54.51, 52.82, 56.73, 57.9, 57.05, 57.02, 60.81, 60.86, 64.82, 77.5, 79.56, 73.56, 75.75]
}

data = pd.DataFrame(data_dict)

plt.style.use('seaborn-darkgrid')

# Scatter plot of Exchange Rate vs Inflation Rate
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data["Exchange Rate"], y=data["Inflation Rate"])
plt.title("Scatter Plot of Exchange Rate vs Inflation Rate")
plt.xlabel("Exchange Rate (1 USD to PKR)")
plt.ylabel("Inflation Rate (%)")
plt.grid(True)
plt.show()

# Scatter plot of Exchange Rate vs Central Government Debt
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data["Exchange Rate"], y=data["Central Government Debt"])
plt.title("Scatter Plot of Exchange Rate vs Central Government Debt")
plt.xlabel("Exchange Rate (1 USD to PKR)")
plt.ylabel("Central Government Debt (%)")
plt.grid(True)
plt.show()

# Perform linear regression for Exchange Rate vs Inflation Rate
X_inflation = data["Exchange Rate"].values.reshape(-1, 1)
y_inflation = data["Inflation Rate"].values

regression_inflation = LinearRegression()
regression_inflation.fit(X_inflation, y_inflation)

# Perform linear regression for Exchange Rate vs Central Government Debt
X_debt = data["Exchange Rate"].values.reshape(-1, 1)
y_debt = data["Central Government Debt"].values

regression_debt = LinearRegression()
regression_debt.fit(X_debt, y_debt)

# Plot linear regression results for Exchange Rate vs Inflation Rate
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data["Exchange Rate"], y=data["Inflation Rate"])
plt.plot(data["Exchange Rate"], regression_inflation.predict(X_inflation), color='red')
plt.title("Linear Regression of Exchange Rate vs Inflation Rate")
plt.xlabel("Exchange Rate (1 USD to PKR)")
plt.ylabel("Inflation Rate (%)")
plt.grid(True)
plt.show()

# Plot linear regression results for Exchange Rate vs Central Government Debt
plt.figure(figsize=(10, 5))
sns.scatterplot(x=data["Exchange Rate"], y=data["Central Government Debt"])
plt.plot(data["Exchange Rate"], regression_debt.predict(X_debt), color='red')
plt.title("Linear Regression of Exchange Rate vs Central Government Debt")
plt.xlabel("Exchange Rate (1 USD to PKR)")
plt.ylabel("Central Government Debt (%)")
plt.grid(True)
plt.show()

# Calculate and display correlations
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

print("Correlation Matrix:")
print(correlation_matrix)