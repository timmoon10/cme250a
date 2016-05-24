#!/usr/bin/python
import h2o
from h2o.grid import H2OGridSearch
from h2o.estimators.random_forest import H2ORandomForestEstimator

# Initialize H2O
h2o.init()

# Load data from files
data_inputs  = h2o.import_file("train_inputs.csv")
data_outputs = h2o.import_file("train_outputs.csv")
test_inputs  = h2o.import_file("test.csv")

# Remove unnecessary column
data_inputs = data_inputs.drop(0) # Drop column number
test_inputs = test_inputs.drop(0) # Drop column number
data_inputs = data_inputs.drop(8) # Drop area code
test_inputs = test_inputs.drop(8) # Drop area code
data_outputs = data_outputs.drop(0) # Drop column number
data_outputs = data_outputs.drop(0) # Drop sample ID

# Rename columns
feature_names = ["Id", "Car", "Transport", "Tickets", "News", "Height",
                 "Shoe", "Courses", "Dates", "Weddings",
                 "Dinner", "Coffee", "Alcohol", "Sleep", "Vote", "Job",
                 "US", "Credit", "Vaccine", "Phone age", "Smart phone age", 
                 "Phone OS", "Pet", "Field"]
for i, name in enumerate(feature_names):
    data_inputs.set_name(i, name)
    test_inputs.set_name(i, name)
output_names = ["Exercise", "Age", "Laptop"]
for i, name in enumerate(output_names):
    data_outputs.set_name(i, name)
data_outputs["Laptop"] = data_outputs["Laptop"].asfactor()

# Merge input and output data frames
data = data_inputs.cbind(data_outputs)

# Impute missing values
data.impute(-1)
test_inputs.impute(-1)

# Generate train set and validation set
[train, val] = data.split_frame([0.7])

# Train models
input_indices = range(1,len(feature_names))
exercise_index = len(feature_names)
age_index = len(feature_names)+1
laptop_index = len(feature_names)+2
hyper_parameters = {'ntrees':[4,8,16,32], 'max_depth':[2,3,4,5]}
grid_exercise = H2OGridSearch(H2ORandomForestEstimator,
                              hyper_params=hyper_parameters)
grid_age = H2OGridSearch(H2ORandomForestEstimator,
                         hyper_params=hyper_parameters)
grid_laptop = H2OGridSearch(H2ORandomForestEstimator,
                            hyper_params=hyper_parameters)
grid_exercise.train(x=input_indices, y=exercise_index,
                    training_frame=train, validation_frame=val)
grid_age.train(x=input_indices, y=age_index,
               training_frame=train, validation_frame=val)
grid_laptop.train(x=input_indices, y=laptop_index,
                  training_frame=train, validation_frame=val)
mse = float("inf")
model_exercise = None
for model in grid_exercise:
    if model.mse(valid=True) < mse:
        mse = model.mse(valid=True)
        model_exercise = model
mse = float("inf")
model_age = None
for model in grid_age:
    if model.mse(valid=True) < mse:
        mse = model.mse(valid=True)
        model_age = model
mse = float("inf")
model_laptop = None
for model in grid_laptop:
    if model.mse(valid=True) < mse:
        mse = model.mse(valid=True)
        model_laptop = model
print "==== EXERCISE MODEL ===="
print model_exercise
print "==== AGE MODEL ===="
print model_age
print "==== LAPTOP MODEL ===="
print model_laptop

# Predict labels of test set
test_exercise = model_exercise.predict(test_inputs[:,1:])
test_age = model_age.predict(test_inputs[:,1:])
test_laptop = model_laptop.predict(test_inputs[:,1:])
test_exercise.set_name(0, "Exercise")
test_age.set_name(0, "Age")
test_laptop.set_name(0, "Laptop")
test_outputs = test_inputs["Id"].cbind(test_exercise["Exercise"])
test_outputs = test_outputs.cbind(test_age["Age"])
test_outputs = test_outputs.cbind(test_laptop["Laptop"].asnumeric())
h2o.export_file(test_outputs, "test_outputs.csv", True)

# Performance
print model_exercise.model_performance(val)
print model_age.model_performance(val)
print model_laptop.model_performance(val)
