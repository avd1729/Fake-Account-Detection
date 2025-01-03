import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "app.log")), 
        logging.StreamHandler()
    ]
)


real_users_df = None
fake_users_df = None

def extract_data_from_csv():
    """
    Extract data from CSV files and store it in global DataFrames.
    """
    global real_users_df, fake_users_df

    real_users = 'data/users.csv' 
    fake_users = 'data/fake_users.csv'

    real_users_df = pd.read_csv(real_users)
    fake_users_df = pd.read_csv(fake_users)

    logging.info("Extracted data from CSV files.")

def transform_data():
    """
    Transform data to necessary structure.
    """
    logging.info("Transforming data (if needed). Currently, no transformations are applied.")

def load_data_to_mongodb():
    """
    Load data into MongoDB collections.
    """
    global real_users_df, fake_users_df

    if real_users_df is None or fake_users_df is None:
        logging.error("DataFrames are empty. Run extract_data_from_csv() first.")
        raise ValueError("No data to load. Extract the data first.")
    
    load_dotenv()
    connection_string = os.getenv("MONGO_CONN_STRING")

    if not connection_string:
        logging.error("MongoDB connection string is not set in environment variables.")
        raise ValueError("Missing MongoDB connection string.")

    client = MongoClient(connection_string)
    db = client['fake_account_data'] 
    
    try:
         real_users_collection = db['real_users']
         fake_users_collection = db['fake_users']
    
         real_users_collection.drop()
         fake_users_collection.drop()
         logging.info("Dropped existing MongoDB collections.")

         real_users_collection.insert_many(real_users_df.to_dict('records'))
         fake_users_collection.insert_many(fake_users_df.to_dict('records'))
         logging.info("Data loaded successfully into MongoDB.")

    except Exception as e:
        logging.error(f"Error loading data into MongoDB: {e}")
        raise

    print("Data loaded successfully!")

# extract_data_from_csv()
# transform_data()
# load_data_to_mongodb()