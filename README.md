# kaggle-titanic-REST-API
Example of doing the Kaggle Titanic problem and running the model as a REST API. The main.py file reads the csv data, performs the machine learning and saves the relevant parameters used to run the server afterwards. The file flask_REST.py then reads the parameters saved by the main.py file and runs the REST API server. 
If the final line in the main.py file is uncommented, the server will be run by running that file alone (note the absence od "if name == main" line in the  flask_REST.py file).
