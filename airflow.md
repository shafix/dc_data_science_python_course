DATA CAMP! Apache airflow!


# Running a DAG task
```
airflow run <dag_id> <task_id> <start_date>
```

# Running a full DAG
```
airflow trigger_dag -e <date> <dag_id>
```

# Listing available DAGs
```
airflow list_dags
```


# Defining a DAG
```
from airflow.models import DAG
from datetime import datetime
default_arguments = {
	'owner': '...',
	'email': '...', # email for alerting
	'start_date': datetime(2020,1,20) 
}
etl_dag = DAG( 'etl_workflow', default_args=default_arguments )
```
# start_date - earliest date when the DAG could be run
# end_date - latest date when the DAG can be run
# max_tries - how many attempts to make
# schedule_interval - how often to run, can use either airflow presets like @hourly @daily or cron definitions like "0 12 * * *" or "0,15,30,45 * * * *"
# retry_delay - how long to wait before retrying




# Task examples (instances of operators, assigned to a variable in python)
# Bash operator
```
from airflow.operators.bash_operator import BashOperator
var_1 = BashOperator(
	task_id='bash_example',
	bash_command='echo "Example!"',
	dag=ml_dag
)
var_2 = BashOperator(
	task_id='bash_script_example',
	bash_command='runcleanup.sh"',
	dag=ml_dag
)
```



# Task dependencies ( upstread - must be completed first/before, downstream - should be completed afterwards)
>> - upstream
<< - downstream
```
task1 = ...
task2 = ...
task1 >> task2 # run task1 before task2
task3 = ...
task1 >> task2 << task3 # run task1 and task3 before running task2
# or ...
task3 >> task2
task3 >> task2
```


# Python operator
# Exectues a function or .py file, can pass arguments/env_variables to the python code 
```
from airflow.operators.python_operator import PythonOperator
def sleep(length_of_time):
	time.sleep(length_of_time);
sleep_task = PythonOperator(
	task_id='sleep',
	python_callable=sleep,
	op_kwargs={'length_of_time': 5}
	dag=example_dag
)
```


```
# Email operator
from airflow.operators.email_operator import EmailOperator
email_task = EmailOperator(
	task_id='email_sales_report',
	to='sales_manager@.."
	subject='Automated Sales Report',
	html_content='Attached is the latest sales report',
	files='latest_sales.xlsx',
	dag=example_dag
)
```





# Sensor operators - get's triggered by a certain condition
# airflow.sensors.base_sensor_operator
# mode - how to check for the condition : mode='poke' - default, run repeatedly , mode='reschedule' - give up task slot and try again later
# poke_interval - how often to check
# timeout - how long to wait until failing 

# File sensor from airflow.contrib.sensors - checks for the existance of a file at a certain location
```
from airflow.contrib.sensors.file_sensor import FileSensor
file_sensor_task = FileSensor(
	task_id='file_sense',
	filepath='salesdata.csv',
	poke_interval=300,
	dag=sales_report_dag
)
init_sales_cleanup >> file_sensor_task >> generate_report
```

# Other sensor types:
```
- ExternalTaskSensor - waits for a task in another DAG to complete
- HttpSensor - requests URL and checks for content
- SqlSensor - runs SQL query and check for content
```

# When to use sensors:
```
- Uncertain it will be true
- If failure not immediatelly desired
- Add task repetition without loops
```






# Airflow executors (worker)
```
- Executors run tasks 
- Different executors may handle running the tasks differently
- Examples : SequentialExecutor, LocalExecutor, CeleryExecutor
- SequentialExecutor - default, runs one task a time, good for debugging, not recommended for production
- LocalExecutor - runs on a single system, treats tasks as processes, parallelism possible, can utilize all resources of a given host system
- Celery - general queueing system written in python, allows multiple systems to communicate as a basic cluster
- CeleryExecutor - Uses a Celery backend as task manager, multiple worker systems can be defined and orchestrated, easy to scale, difficult to setup and configure, extremely powerful.
```

# How to determine the current executor of the airflow system:
```
- Check airflow.cfg file, and check the "executor=" attribute
- Run "airflow list_dags" and check the top line INFO
```



# Running the scheduler if it's not running
```
airflow scheduler
```




# SLAs and SLA misses / fails:
```
- Can determine the SLA for a specific task by adding a "sla=timedelta(seconds=30)" attribute to the task itself.
- When defining a DAG add a 'sla': timedelta(minutes=20) attribute to the default_args dictionary.
- Email reporting can be define in the DAG default_args dictionary : { 'email': [list of emails], 'email_on_failure': True, 'email_on_retry': False, 'email_on_success': True }
```


# Timedelta object:
```
- from datetime import timedelta
- Can take weeks, days, hours, minutes as arguments
- Examples : timedelta(seconds=30) ; timedelta(days=4, hours=10, minutes=20,seconds=30)
```




# Templates with Jinja
```
templated_command="""
	echo "Reading {{ params.filename }}"
"""
t1 = BashOperator(
	task_id='template_task',
	bash_command=templated_command,
	params={'filename': 'file1.txt'},
	dag=example_dag
)

templated_command="""
	<% for filename in params.filenames %>
	echo "Reading {{filename}}"
	<% endfor %>
"""
t1 = BashOperator(
	task_id='template_task',
	bash_command=templated_command,
	params={'filenames': ['file1.txt','file2.txt']},
	dag=example_dag
)
```


# Some built-in runtime variables:
```
# {{ ds }} - execution date in YYYY-MM-DD
# {{ ds_nodash }} - execution date in YYYYMMDD
# {{ prev_ds }} - previous execution date in YYYY-MM-DD
# {{ prev_ds_nodash }} - previous execution date in YYYYMMDD
# {{ dag }} - the DAG object itself
# {{ conf }} - current airflow config object
# {{ macros }} - access the Airflow macros package that has useful objects and methods for templates, for example macros.datetime, macros.timedelta, macros.ds_add('2020-04-15', 5)
```



# Branching - BranchPythonOperator
```
from airflow.operators.python_operator import BranchPythonOperator
def branch_test(**kwargs):
	if int( kwargs['ds_nodash'] ) % 2 == 0:
		return 'even_day_task'
	else:
		return 'odd_day_task'

branch_task = BranchPythonOperator(
	task_id='branch_task',
	dag=example_dag,
	python_callable=branch_test,
	provide_context=True
)

start_task >> branch_task >> even_day_task
branch_task >> odd_day_task
```
