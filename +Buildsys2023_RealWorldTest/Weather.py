#%%
# Getting Hourly data from a library
# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly

# Set time period
start = datetime(2023, 6, 26)
end = datetime(2023, 6, 27)

# Create Point for Vancouver, BC
location = Point(37.229572, -80.413940)

# Get daily data for 2018
data = Hourly(location, start, end)
data = data.fetch()
print(data)

# Plot line chart including average, minimum and maximum temperature
data.plot(y=['temp'])
plt.show()

#%% Getting current temperature
import requests
from datetime import datetime

API_KEY = '773d26c7dc9c8cc08f663e133833c7de'
CITY_NAME = 'Blacksburg,US'

# Get the current date and time
current_date = datetime.now().strftime('%Y-%m-%d')

# Create the API request URL
url = f'https://api.openweathermap.org/data/2.5/weather?q={CITY_NAME}&appid={API_KEY}'

# Send the API request
response = requests.get(url)
data = response.json()

# Check if the request was successful
if response.status_code == 200:
    temperature = (data['main']['temp'] - 273.15) * 1.8 + 32
    print(f"The current temperature in {CITY_NAME} is {temperature} Kelvin.")
else:
    print(f"Error: {data['message']}")

#%% Getting future temperature
import requests
import json
from datetime import datetime, timedelta

API_KEY = '773d26c7dc9c8cc08f663e133833c7de'
CITY_NAME = 'Blacksburg'
BASE_URL = 'http://api.openweathermap.org/data/2.5/forecast'

# Get the current timestamp and the timestamp 15 minutes from now
current_time = datetime.now()
future_time = current_time + timedelta(minutes=15)
future_timestamp = int(future_time.timestamp())

# Make the API request
params = {
    'q': CITY_NAME,
    'appid': API_KEY,
    'units': 'metric',
}
response = requests.get(BASE_URL, params=params)
data = json.loads(response.text)

# Find the temperature closest to the future timestamp
closest_temp = None
for forecast in data['list']:
    forecast_time = int(forecast['dt'])
    if forecast_time >= future_timestamp:
        closest_temp = forecast['main']['temp'] * 1.8 + 32
        break

if closest_temp is not None:
    print(f"The temperature in {CITY_NAME} in the next 15 minutes will be {closest_temp}°C.")
else:
    print("Unable to retrieve temperature information.")




