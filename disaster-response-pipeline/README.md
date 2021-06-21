# Disaster Response Pipeline Project

## 1. Description of files in this folder
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


## 2. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/MESSAGES.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/MESSAGES.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

## 3. Summary 

