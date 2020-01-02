import numpy as np
import pandas as pd

# Creating a Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Creating a DataFrame
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
print(df)
