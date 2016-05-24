#!/usr/bin/python
import csv
import h2o
from h2o.grid import H2OGridSearch
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Initialize H2O
h2o.init()

# Load headers from file
header_file = '/Users/moon/Downloads/Xheader.csv'
with open(header_file, 'r') as f:
    headers = csv.reader(f).next()

# Load data from files
data_files = ['/Users/moon/Downloads/X1940.csv', '/Users/moon/Downloads/X2016.csv']
data = h2o.import_file(data_files)

# Rename data frame columns
for col, name in enumerate(headers):
    data.set_name(col, name)
        
print data[:4,:]

# Generate train set and validation set
[train, val, test] = data.split_frame([0.7, 0.2])

# Train models
feature_list = ['Year', 'Station']
model = H2ORandomForestEstimator()
model.train(feature_list, 'temp', train, validation_frame=val)

# Predict labels of test set
pred = model.predict(test)

# Performance
print model.model_performance(val)
