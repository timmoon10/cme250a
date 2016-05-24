import csv
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Parameters
header_file = '/home/tym1/Downloads/Xheader.csv'
data_files = ['/home/tym1/Downloads/X1940.csv', '/home/tym1/Downloads/X2016.csv']
model = H2ORandomForestEstimator(ntrees=50, max_depth=20)

# Initialize H2O
h2o.init()

# Load headers from file
with open(header_file, 'r') as f:
    headers = csv.reader(f).next()

# Load data from files
data_raw = h2o.import_file(data_files)

# Rename data frame columns
for col, name in enumerate(headers):
    data_raw.set_name(col, name)

# Combine WBAN and DAVSAT3 station IDs to get unified station IDs
StationIds = 100000*data_raw['Station'] + data_raw['WBAN']
StationIds = StationIds.asfactor()
StationIds.set_name(0, 'StationId')
data_raw = data_raw.cbind(StationIds)

# Obtain date
months = data_raw['MonthDay'] // 100
days = data_raw['MonthDay'] % 100
months.set_name(0, 'Month')
days.set_name(0, 'Day')
data_raw = data_raw.cbind(months)
data_raw = data_raw.cbind(days)

# Consolidate relevant data
data = data_raw['temp']
data = data.cbind(data_raw['StationId'])
data = data.cbind(data_raw['Year'])
data = data.cbind(data_raw['Month'])
data = data.cbind(data_raw['Day'])

# Remove missing data
temps = data['temp']
temps[temps>9999] = float('nan')
data['temp'] = temps
if data.nacnt()[0] != 0:
    print "WARNING: missing temperature detected"

# Generate train set and validation set
[train, val, test] = data.split_frame([0.7, 0.2])

# Train model
feature_list = ['Year', 'Month', 'Day', 'StationId']
model.train(feature_list, 'temp', train, validation_frame=val)

# Determine model performance on test set
pred = model.model_performance(test, True, True)

# Display results
print model
print pred
