import pandas as pd

# Paths to the two CSV files
file1 = '../data/data_without_mouseevents.csv'  # Replace with the path to your first CSV file
file2 = '../data/bots_all.csv'  # Replace with the path to your second CSV file

# Load the CSV files into dataframes
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Append the rows of the second dataframe to the first
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged dataframe to a new CSV file
output_file = "data_without_mouseevents_new1.csv"
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as '{output_file}'")
