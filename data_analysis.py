import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pandas import read_csv
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# Define the path to your dataset
dataset_path = "E:/Py/housing.csv"

# Load the dataset
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(dataset_path, header=None, delimiter=r"\s+", names=column_names)

# Print the first 5 rows of the dataset
print(data.head())






##! statistical  Print the shape of the dataset
print("statistical informatiob about dataset <the shape>")
print(np.shape(data))
print(data.describe())

# Calculate the percentage of outliers in each column
for k, v in data.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))

# Remove outliers from the 'MEDV' column
data = data[~(data['MEDV'] >= 50.0)]
print(np.shape(data))


selected_columns = ['TAX', 'RM', 'LSTAT', 'DIS', 'MEDV']
selected_data = data[selected_columns]

print(selected_data.head(50))



# Plot histograms for each column
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
axs = axs.flatten()
for i, (col, values) in enumerate(data.items()):
    sns.histplot(values, ax=axs[i], kde=True)  # Set kde parameter to True
    axs[i].set_title(col)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

##! create adjacency to see the value of correlation coefficients by using heatmap.
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr().abs(), annot=True)
plt.show()

##! visualization of correlation coefficients

# Let's scale the columns before plotting them against MEDV
min_max_scaler = MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:,column_sels]
y = data['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
axs = axs.flatten()

# First row for RM', 'LSTAT', 'DIS', and 'TAX'
for i, k in enumerate(['RM', 'LSTAT', 'DIS', 'TAX']):
    sns.regplot(y=y, x=x[k], ax=axs[i], color='green')
    axs[i].set_title(k)

# Second row remain data
for i, k in enumerate(['INDUS', 'NOX', 'PTRATIO', 'AGE']):
    sns.regplot(y=y, x=x[k], ax=axs[i+4], color='red')
    axs[i+4].set_title(k)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

print(len(data))


##!  save cleand data 

# # Define the path to save the dataset
# save_path = "C:/Users/pc/regression/housing_cleaned.csv"

# # Save the dataset to a CSV file
# data.to_csv(save_path, index=False)

# # Print a confirmation message
# print("Dataset saved successfully at:", save_path)






