import pandas as pd

# Make sure the path is correct
data = pd.read_csv("Fake.csv") 
data = pd.read_csv("True.csv") 
print(data.head())
