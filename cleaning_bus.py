import pandas as pd

#subway_data = pd.read_csv('subway_data.csv')

# this will drop any data that Gpt wasnt able to obtain coordinates for, those subway station did not exist or mispelled
#clean_subway_data = subway_data.dropna(subset=['Latitude', 'Longitude'])

#clean_subway_data.to_csv("clean_subway_data.csv")

#print(clean_subway_data.head(20))


### Below is the tester code to edit csv and save csv as new files
#test = pd.read_csv('test_data.csv')

#clean_subway_data = test.dropna(subset=['Latitude', 'Longitude'])

#clean_subway_data.to_csv("clean_test_data.csv", index=False)


# step 1, read the data and change the date format to YYYY-MM-DD, and 


bus_data = pd.read_csv('bus_data.csv')
bus_data["Date"] = pd.to_datetime(bus_data["Date"], format="%d-%b-%y").dt.strftime("%Y-%m-%d")

bus_data = bus_data.dropna()


# change time from string to time data type



# Clean the Time column (remove any unexpected characters)
bus_data["Time"] = bus_data["Time"].astype(str).str.strip()

# Ensure 'Time' column is properly formatted by extracting HH:MM only
bus_data["Time"] = bus_data["Time"].str.extract(r'(\d{1,2}:\d{2})')[0]

# Convert 'Time' column to datetime format and extract the time part
bus_data["Time"] = pd.to_datetime(bus_data["Time"], format="%H:%M", errors='coerce').dt.time


# step 2, remove all 0,0 min gap and min delays in the datasets

bus_data = bus_data[bus_data["Min Delay"] != 0]
bus_data = bus_data[bus_data["Min Gap"] != 0]


# step 3, add rush hour index if the recorded delay is during rush hour, 
# rush hour by definition is 6:30-10:00 am and 3:30 to 7:30 pm

# Define rush hour time ranges
morning_start = pd.to_datetime("06:30", format="%H:%M").time()
morning_end = pd.to_datetime("10:00", format="%H:%M").time()
evening_start = pd.to_datetime("15:30", format="%H:%M").time()
evening_end = pd.to_datetime("19:00", format="%H:%M").time()

# Function to assign rush hour index
def rush_hour_index(time):
    if morning_start <= time <= morning_end or evening_start <= time <= evening_end:
        return 1  # Rush Hour
    return 0  # Non-Rush Hour

# Apply function to create a new column
bus_data["Rush Hour Index"] = bus_data["Time"].apply(rush_hour_index)

print(bus_data.head(5))

# step 4 merge weather data

weather_data = pd.read_csv("weather_data.csv")

weather_data = weather_data[['Date/Time', 'Mean Temp (°C)', 'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)', 'Snow on Grnd (cm)', 'Spd of Max Gust (km/h)']]

print(weather_data.head(5))


# Merge on the 'date' column (inner join to keep only matching dates)

weather_data.rename(columns={'Date/Time': 'Date'}, inplace=True)

merged_data = pd.merge(bus_data, weather_data, on='Date', how='inner')

print(merged_data.head(5))

merged_data.to_csv('bus_model_data.csv')

# clean weather data to remove unnessory columns, 
# keep Snow on Grnd, Speed of Max gust, Percipitation, Total Snow, mean temp