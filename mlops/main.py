import os
from utils.fake_account_detector import FakeAccountDetector
from utils.load_data import export_to_csv
from pymongo import MongoClient
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

    
load_dotenv()
connection_string = os.getenv("MONGO_CONN_STRING")

if not connection_string:
    raise ValueError("Missing MongoDB connection string.")

client = MongoClient(connection_string)
db = client['fake_account_data'] 

real_users = db['real_users']
fake_users = db['fake_users']


export_to_csv(real_users, 'data/real_users.csv')
export_to_csv(fake_users, 'data/fake_users.csv')


detector = FakeAccountDetector()


X, y = detector.load_and_preprocess_data('data/real_users.csv', 'data/fake_users.csv')
results = detector.train(X, y)
    

for metric, value in results.items():
   print(f"{metric}: {value:.4f}")

detector.plot_feature_importance()
detector.save_models("C:/Users/Aravind/fake-data-detection/mlops/models")