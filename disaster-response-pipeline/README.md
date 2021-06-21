# Disaster Response Pipeline Project

## 1. Summary 
This project aims to create a web application which utilizes a machine learning classification model to classify disaster Twitter messages. For more details, please see sections 2 and 3. 

### a. Model performance metrics

From the 2 csv files, a [Random Forest Classification model](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) is trained. 

GridSearchCV is used to obtain the optimal parameters. Please see the `ML Pipeline Preparation.ipynb` file for more details.

The performance metrics are as follows: 

![Model performance](https://github.com/AhmadHatziq/data-science-nanodegree/blob/main/disaster-response-pipeline/images/model_performance.png)

### b. Web app visualizations

A screenshot of the visualizations are as follows:

![Visualizations](https://github.com/AhmadHatziq/data-science-nanodegree/blob/main/disaster-response-pipeline/images/flask_visualizations.png)


## 2. Description of files in this folder
The directory of the folder is as follows:
* /app 
	* run.py - Python file used to launch Flask web-application. 
	* templates - HTML templates used in Flask.
* /data
	* MESSAGES.db - Database generated from source csv files. 
	* disaster_categories.csv - Source csv file containing categories information. 
	* disaster_messages.csv - Source csv file containing messages information. 
	* process_data.py - Python file used for the ETL process. Extracts & cleans data from `disaster_categories.csv` and `disaster_messages.csv` and outputs into the database file `MESSAGES.db`.
* /models: 
	* train_classifier.py - Python file used to output a classifier obtained by training from  `MESSAGES.db`.
	* classifier.pkl - Classification model. Omitted as file is quite large. 
* /notebooks: 
	* ETL Pipeline Preperation.ipynb - Python notebook used for ETL process. 
	* ML Pipeline Preperation.ipynb - Python notebook used for ML training process. 
* `README.md` - this file.


## 3. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/MESSAGES.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/MESSAGES.db models/classifier.pkl`

2. Run the following command in the app's directory to run the Flask web app.
    `python app/run.py`



