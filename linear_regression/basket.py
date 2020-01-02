import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.close('all')

csv = pd.read_csv(os.path.dirname(__file__) + '/../dataset/cbb.csv')
df = pd.DataFrame(csv.loc[:, ['EFG_O', 'W']])

print(df.shape)
print(df.describe())
df.plot.scatter(x='EFG_O', y='W')
plt.title('Field goal % impact on number of wins.')
plt.xlabel('Field goal %')
plt.ylabel('Wins')
plt.show()

attribute = df['EFG_O'].values.reshape(-1, 1)
label = df['W'].values.reshape(-1, 1)

attr_train, attr_test, label_train, label_test = train_test_split(attribute, label, test_size=0.5, random_state=0)

model = LinearRegression()
model.fit(attr_train, label_train)

label_predict = model.predict(attr_test)

df = pd.DataFrame({'Actual': label_test.flatten(), 'Predicted': label_predict.flatten()})

# print(df)

print('Mean Absolute Error:', metrics.mean_absolute_error(label_test, label_predict))
print('Mean Squared Error:', metrics.mean_squared_error(label_test, label_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(label_test, label_predict)))

plt.scatter(attr_test, label_test,  color='gray')
plt.plot(attr_test, label_predict, color='red', linewidth=2)
plt.show()
