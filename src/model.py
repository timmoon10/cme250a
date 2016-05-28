import csv
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Parameters
header_file = '/home/tym1/Downloads/Xheader.csv'
data_files = ['/home/tym1/Downloads/X1940.csv', '/home/tym1/Downloads/X2016.csv']
model = H2ORandomForestEstimator()

# Initialize H2O
h2o.init()

# Load data from files
data = h2o.import_file(data_files)
with open(header_file, 'r') as f:
    headers = csv.reader(f).next()
for col, name in enumerate(headers):
    data.set_name(col, name)

# Convert Boolean data to categorical
data['fog'] = data['fog'].asfactor()
data['rain'] = data['rain'].asfactor()
data['snow'] = data['snow'].asfactor()
data['hail'] = data['hail'].asfactor()
data['thunder'] = data['thunder'].asfactor()
data['tornado'] = data['tornado'].asfactor()

# Delete unnecessary columns
data = data.drop('temp cnt')
data = data.drop('dewpoint cnt')
data = data.drop('sea cnt')
data = data.drop('stat cnt')
data = data.drop('visi cnt')
data = data.drop('wind speed cnt')
data = data.drop('*is hourly max')
data = data.drop('*is hourly min')

# Combine WBAN and DAVSAT3 station IDs to get unified station IDs
StationIds = 100000*data['Station'] + data['WBAN']
StationIds = StationIds.asfactor()
StationIds.set_name(0, 'StationId')
data = data.cbind(StationIds)
data = data.drop('Station')
data = data.drop('WBAN')

# Obtain date
months = data['MonthDay'] // 100
days = data['MonthDay'] % 100
months.set_name(0, 'Month')
days.set_name(0, 'Day')
data = data.cbind(months)
data = data.cbind(days)
data = data.drop('MonthDay')

# Add next day temperature
today = h2o.H2OFrame.mktime(data['Year'], months-1, days-1, 12)
today = h2o.H2OFrame._expr(today)
yesterday = today - 86400000
tomorrow_data = data['StationId']
tomorrow_data = tomorrow_data.cbind(data['temp'])
tomorrow_data = tomorrow_data.cbind(yesterday.year())
tomorrow_data = tomorrow_data.cbind(yesterday.month())
tomorrow_data = tomorrow_data.cbind(yesterday.day())
tomorrow_data.set_name(1, 'tomorrow_temp')
tomorrow_data.set_name(2, 'Year')
tomorrow_data.set_name(3, 'Month')
tomorrow_data.set_name(4, 'Day')
data = data.merge(tomorrow_data)

# Remove entries with missing temperature data
data[data['temp']>9999,'temp'] = None
data[data['tomorrow_temp']>9999,'tomorrow_temp'] = None
data = data.na_omit()

# Remove missing data
data[data['dewpoint']>9999,'dewpoint'] = None
data[data['sea level pres']>9999,'sea level pres'] = None
data[data['station pres']>9999,'station pres'] = None
data[data['visibility']>999,'visibility'] = None
data[data['mean wind speed']>999,'mean wind speed'] = None
data[data['max wind speed']>999,'max wind speed'] = None
data[data['gust speed']>999,'gust speed'] = None
data[data['max temp']>9999,'max temp']= None
data[data['min temp']>9999,'min temp']= None
data[data['precipitation']>99,'precipitation'] = 0
data[data['snow depth']>999,'snow depth'] = 0

# Generate train set and validation set
[train, val, test] = data.split_frame([0.7, 0.2])

# Train model
feature_list = list(data.names)
feature_list.remove('tomorrow_temp')
model.train(feature_list, 'tomorrow_temp', train, validation_frame=val)

# Determine model performance on test set
pred = model.model_performance(test, True, True)

# Display results
print model
print pred
