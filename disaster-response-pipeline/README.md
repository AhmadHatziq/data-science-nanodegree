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
	* process_data.py - Python file used for the ETL process. Extracts & cleans data from `disaster_categories.csv` and `disaster_messages.csv` and outputs into the database `MESSAGES.db`.
* `create_cluster.py`: IaC file. Used to create the Redshift database and store login credentials. 
* `delete_cluster.py`: IaC file. Used to delete the Redshift database. 
* `dwh.cfg`: Configuration file used for storing AWS and Redshift credentials.
* `source_file_to_parquet.py`: Loads the raw data files from S3 into Spark, performs data cleaning and writes parquet files back into S3.
* `parquet_to_redshift.py`: Loads the parquet files from S3 into the Redshift database.
* `sql_statements.py`: SQL statements used in the ETL process.
* `data_quality.py`: Performs data quality checks.
* `etl.py`: Executes the whole ETL process, using all the above files except for `create_cluster.py` and `delete_cluster.py`.
* `README.md` - this file
* `data_dictionary.md` - Data dictionary of final tables
* `capstone_project_report.ipynb` - Final report of this project

## 2. Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
