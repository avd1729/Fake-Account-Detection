from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from utils.util import extract_data_from_csv, transform_data, load_data_to_mongodb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting dag...")

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

with DAG(dag_id='etl_pipeline',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dags:
    
    @task
    def extract():
       try:
           extract_data_from_csv()
       except:
           logging.error("Data Extraction failed!.")

    @task
    def transform():
        try:
           transform_data()
        except:
           logging.error("Data Transformation failed!.")


    @task
    def load():
        try:
           load_data_to_mongodb()
        except:
           logging.error("Data Loading failed!.")

    extract() >> transform() >> load()