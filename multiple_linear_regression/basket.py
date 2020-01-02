import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Close all opened plots
plt.close('all')

data_set = pd.read_csv(os.path.dirname(__file__) + '/../dataset/cbb.csv')
# Number of rows and cols in a data set
print(data_set.shape)
# Statistical details of the data set
print(data_set.describe())
# Check empty values
print(data_set.isnull().any())

# Cleanup data
data_set.drop(inplace=True, columns=['TEAM', 'CONF', 'POSTSEASON', 'SEED', 'YEAR'])
# Number of rows and cols in a data set
print(data_set.shape)
# Statistical details of the data set
print(data_set.describe())
# Check empty values
print(data_set.isnull().any())

# Take all attributes
columns = data_set.columns.values.tolist()
# Except Wins
columns.remove('W')
# And Games Played
columns.remove('G')
attributes = data_set[columns]
label = data_set['W']

# Separate training set and test set
attr_train, attr_test, label_train, label_test = train_test_split(attributes, label, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(attr_train, label_train)

# Check coefficients
coefficients = pd.DataFrame(model.coef_, attributes.columns, columns=['Coefficient'])
print(coefficients)

# Make predictions
label_predict = model.predict(attr_test)
df = pd.DataFrame({'Actual': label_test, 'Predicted': label_predict})

df_head = df.head(20)
print(df_head)

# Plot the difference between
df_head.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# How did it go?
print('Mean Absolute Error:', metrics.mean_absolute_error(label_test, label_predict))
print('Mean Squared Error:', metrics.mean_squared_error(label_test, label_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(label_test, label_predict)))
