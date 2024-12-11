from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from kafka_producer import initiate_stream  

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fmri_pipeline',
    default_args=default_args,
    description='An fMRI data pipeline with Kafka, Spark, and PyTorch',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# Kafka Producer Task
kafka_producer_task = BashOperator(
    task_id='kafka_producer',
    bash_command='python /path/to/kafka_producer.py',
    dag=dag,
)

# Spark Processing Task
spark_processing_task = BashOperator(
    task_id='spark_processing',
    bash_command='spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2 /path/to/spark_script.py',
    dag=dag,
)

# PyTorch Training Task
def train_model():
    import torch
    # Add your training logic here from the provided model script

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# kafka_producer_task >> spark_processing_task >> train_task

# Creating the DAG with its configuration
with DAG(
    'name_stream_dag',  # Renamed for uniqueness
    default_args=DAG_DEFAULT_ARGS,
    schedule_interval='0 1 * * *',
    catchup=False,
    description='Stream random names to Kafka topic',
    max_active_runs=1
) as dag:
    
    # Defining the data streaming task using PythonOperator
    kafka_stream_task = PythonOperator(
        task_id='stream_to_kafka_task', 
        python_callable=initiate_stream,
        dag=dag
    )

    kafka_stream_task
