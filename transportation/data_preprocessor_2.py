'''
# Installation
pip install pandas
pip install ace-tools
'''

'''
Data Loading & Preprocessing
# Latitude,Longitude columns -> added using ChatGPT
# Convert date/time fields to appropriate formats.
# Merge relevant datasets (e.g., subway delay codes with subway data).
# Handle missing values and ensure consistency.


''' 
import pandas as pd

# File paths
subway_data_path = "data/subway-data.csv"
subway_delay_codes_path = "data/subway-delay-codes.csv"

# Load datasets
subway_data = pd.read_csv(subway_data_path)
subway_delay_codes = pd.read_csv(subway_delay_codes_path)

# Display first few rows of both datasets
print(subway_data.head())
print(subway_delay_codes.head())


# Convert 'Date' and 'Time' to datetime format
subway_data['Datetime'] = pd.to_datetime(subway_data['Date'] + ' ' + subway_data['Time'], format='%Y/%m/%d %H:%M')
# Extract useful time-based features
subway_data['Hour'] = subway_data['Datetime'].dt.hour
subway_data['DayOfWeek'] = subway_data['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
subway_data['Month'] = subway_data['Datetime'].dt.month


# Rename columns in subway-delay-codes dataset to match expected names
subway_delay_codes.rename(columns={'SUB RMENU CODE': 'Code', 'CODE DESCRIPTION': 'DESCRIPTION'}, inplace=True)
# Convert 'Code' in subway_delay_codes to string for consistency
subway_delay_codes['Code'] = subway_delay_codes['Code'].astype(str)
# Merge with subway delay codes to get descriptions
# subway_data = subway_data.merge(subway_delay_codes[['CODE', 'DESCRIPTION']], left_on='Code', right_on='CODE', how='left')
# Merge subway data with delay codes on corrected column names
subway_data = subway_data.merge(subway_delay_codes[['Code', 'DESCRIPTION']], on='Code', how='left')

# Drop redundant columns after merging
subway_data.drop(columns=['Date', 'Time', 'CODE'], inplace=True)

# Display cleaned dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Processed Subway Data", dataframe=subway_data)




