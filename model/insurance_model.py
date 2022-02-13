from pycaret.datasets import get_data
from pycaret.regression import *

# get default pycaret dataset
data = get_data('insurance')

# setup pycaret and set the target as the charges col
s = setup(data, target = 'charges', session_id = 123)

# create linear regression model
lr = create_model('lr')

# tune linear model for optimum parameters
tuned_lr = tune_model(lr)

# plot tuned model
plot_model(tuned_lr)

# save model
save_model(tuned_lr, 'deploy_tunedlr_02132022')

# load model to test
deploy_tunedlr_02132022 = load_model('deploy_tunedlr_02132022')

# print saved model details
print(deploy_tunedlr_02132022)

# test
# import requests
# url = 'https://pycaret-insurance.herokuapp.com/predict_api'
# pred = requests.post(url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})
# print(pred.json())