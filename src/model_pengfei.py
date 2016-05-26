import csv
import h2o
header_file = '/Users/Pengfei/Documents/data/Xheader.csv'
data_files = ['/Users/Pengfei/Documents/data/X2015.csv', '/Users/Pengfei/Documents/data/X2016.csv']
mixed_files = ['/Users/Pengfei/Documents/data/Xheader.csv','/Users/Pengfei/Documents/data/X2015.csv', '/Users/Pengfei/Documents/data/X2016.csv']

# Initialize H2O
h2o.init()

# Load headers from file
with open(header_file, 'r') as f:
    headers = csv.reader(f).next()

# Load data from files
data_raw = h2o.import_file(mixed_files)

data_raw[data_raw['snow depth']>999,'snow depth'] = None
data_raw[data_raw['precipitation']>99,'precipitation'] = None
data_raw[data_raw['min temp']>9999,'min temp']= None
data_raw[data_raw['max temp']>9999,'max temp']= None
data_raw[data_raw['gust speed']>999,'gust speed'] = None
data_raw[data_raw['max wind speed']>999,'max wind speed'] = None
data_raw[data_raw['mean wind speed']>999,'mean wind speed'] = None
data_raw[data_raw['visibility']>999,'visibility'] = None
data_raw[data_raw['station pres']>9999,'station pres'] = None
data_raw[data_raw['sea level pres']>9999,'sea level pres'] = None
data_raw[data_raw['dewpoint']>9999,'dewpoint'] = None



# for i in range(0,len(data_raw.columns)):
# 	data_raw[i].summary()
