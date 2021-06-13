"""
Created on Sun May  2 12:13:52 2021

@author: Harry Jackson
"""
#Imports
import pandas as pd
import numpy as np

"""
This function reformats the inputted predictions and actuals from wide format to long 

Args:
    predictions (dataframe): This is the nodes predictions for the labels by content hash in the format -[content_hash, node_identifier, adult, suggestive, 
                                                                           violence, visually_disturbing, hate_symbols].
    actuals: (dataframe): This is the actual values for the labels -[content_hash, adult, suggestive, 
                                                                           violence, visually_disturbing, hate_symbols].

Returns (dataframe, dataframe):
    Two dataframes that have been pivoted from wide to long
"""
def wide_to_long(predictions, actuals):
    #turn into long format
    long_predictions = pd.melt(predictions,
                            id_vars=['content_hash','node_identifier'],
                            value_vars=['adult','suggestive','violence','visually_disturbing','hate_symbols'],
                            var_name = "label",
                            value_name = "prediction")
    long_actuals = pd.melt(actuals,
                            id_vars=['content_hash'],
                            value_vars=['adult','suggestive','violence','visually_disturbing','hate_symbols'],
                            var_name = "label",
                            value_name = "actual")
    return(long_predictions, long_actuals)

"""
This function calculates the RMSE of a node. The error is the difference betweent the prediction and the actual values by label.
To calculate this we square the individual errors, take the mean by node/label, then square root the result.
This will give us our error %, by taking this from 100 we can thus calculate our accuracy.
This value does not need standardising as the max of a label is 100 and the min is 0.
Thus the max overall accuracy of a node is 100 and the minimum is 0. 

Args:
    errors (dataframe): A dataframne with cols [node_identifier, label, error]

Returns (dataframe):
    A dataframe with the RMSE accuracy calculated by node/label
"""
def calculate_RMSE(errors):

    errors["squared_error"] = errors["error"]**2
    #calculate the mean of the squared erros by node/label
    results = errors.groupby(["node_identifier", 'label']).mean()
    results['RMSE'] = results['squared_error'] **0.5
    results['accuracy'] = 100 - results['RMSE'] 
    return(results.reset_index())

"""
This function takes the accuracy scores by node and label and will calculate the weighting each node should recieve by normalising these accuracies.


Args:
    results (dataframe): The dataframe to normalise the accuracies across to provide a weighting, in the format - [node_identifier, label, accuracy]

Returns (dataframe):
    A dataframe of weights by node containing the weighting a node should be given in the aggregation. 
"""
def agg_normalise(results):
    weights = results[["node_identifier", "label", "accuracy"]].copy()
    #replace na with 0
    weights['accuracy'] = weights['accuracy'].fillna(0)
    weights['total'] = weights.groupby("label")["accuracy"].transform('sum')
    weights['weighting'] = weights['accuracy']/weights['total']
    weights = weights.drop(['total'], axis = 1)
    return(weights)
    
#Load in training and testing data
prediction_data = pd.read_csv('training_data.csv',dtype={0:'str',1:'str',2:np.float64,3:np.float64,4:np.float64,5:np.float64,6:np.float64})
actual_data = pd.read_csv('testing_data.csv',dtype={0:'str',1:np.float64,2:np.float64,3:np.float64,4:np.float64,5:np.float64})

#Pivot data to long format
long_prediction, long_actual = wide_to_long(prediction_data, actual_data)

#Join actuals to predictions based upon content hash
combined = long_prediction.merge(right=long_actual, how = 'inner', left_on = ['content_hash', 'label'], right_on = ['content_hash', 'label'])
#Calculate error between prediction and actual
combined["error"] = combined["prediction"] - combined["actual"] 
#Calculate root mean squared error from error and actuals
results = calculate_RMSE(combined)
#Normalise RMSE across nodes to return a normalised weighting
weights = agg_normalise(results)
print("WEIGHTS")
print(weights)
#Save to CSV
weights.to_csv("node_weightings.csv")

