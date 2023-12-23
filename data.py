import pandas as pd

fake_csv_path = "news_datasets/Fake.csv"
true_csv_path = "news_datasets/True.csv"

# Read the CSV files
fake_data = pd.read_csv(fake_csv_path)
true_data = pd.read_csv(true_csv_path)
fake_data.head() 
true_data.head()

# Read data
# print("Fake Data:")
# print(fake_data.head())

# print("\nTrue Data:")
# print(true_data.head())
