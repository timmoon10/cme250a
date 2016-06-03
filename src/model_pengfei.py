import csv
import h2o
header_file = '/Users/Pengfei/Documents/data/Xheader.csv'
data_files = ['/Users/Pengfei/Documents/data/X2015.csv', '/Users/Pengfei/Documents/data/X2016.csv']
mixed_files = ['/Users/Pengfei/Documents/data/Xheader.csv','/Users/Pengfei/Documents/data/X2015.csv', '/Users/Pengfei/Documents/data/X2016.csv']
file1 = ['/Users/Pengfei/Documents/data/Xheader.csv','/Users/Pengfei/Documents/data/X1940.csv']
# file2 = ['/Users/Pengfei/Documents/data/Xheader.csv','/Users/Pengfei/Documents/data/X2016.csv']

# Initialize H2O
h2o.init()

# Load data from files
data = h2o.import_file(file1)


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

# Remove entries with missing temperature data
data[data['temp']>9999,'temp'] = None

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

# set chosen feature
feature_list = list(data.names)
feature_list.remove('temp')
feature_list.remove('max temp')
feature_list.remove('min temp')

#specify model detail
gbm_model = h2o.estimators.gbm.H2OGradientBoostingEstimator(model_id='gbm1', distribution='gaussian',nfolds=4)
#train model
gbm_model.train(y = "temp", x = feature_list, training_frame = train, validation_frame = val)
#predict on test data set
predicted_result = gbm_model.predict(test)
#calculate mse for both train and valid data
gbm_model.mse(train=True, valid=True, xval=True)
#plot mse error history for training and validation data
gbm_model.plot()
#report performance of train validation and test ????usage
gbm_model.model_performance(test_data = test, train = True, valid = True)
#list of score history
gbm_model.scoring_history()


#try cross validation framework
glm_model = h2o.estimators.glm.H2OGeneralizedLinearEstimator(model_id='glm1',nfolds=4)
glm_model.train(y = "temp", x = feature_list, training_frame = train, validation_frame = val)
glm_model.mse(train=True, valid=True, xval=True)


#try grid search
from h2o.grid.grid_search import H2OGridSearch
#try grid search, using validation set
hyper_parameters = {'ntrees':[100], 'max_depth':[3,5,7,9,11], 'learn_rate':[0.01,0.05,0.1,0.25]}
gs = H2OGridSearch(h2o.estimators.gbm.H2OGradientBoostingEstimator(distribution='gaussian'), hyper_params=hyper_parameters)
gs.train(y = "temp", x = feature_list, training_frame = train, validation_frame = val)
gs.show() #rank by validation error


#try grid search, using CV
hyper_parameters = {'ntrees':[50], 'max_depth':[3,5,7,10], 'learn_rate':[0.01,0.05,0.25]}
gs = H2OGridSearch(h2o.estimators.gbm.H2OGradientBoostingEstimator(distribution='gaussian',nfolds=3), hyper_params=hyper_parameters)
gs.train(y = "temp", x = feature_list, training_frame = train, validation_frame = val)
gs.show() #rank by CV error








