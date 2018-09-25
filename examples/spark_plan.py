import os
from ektelo import data as ektelo_data
from ektelo import spark as ektelo_spark
from pyspark.sql import SparkSession
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')

filename =  os.path.join(CSV_PATH, 'cps.csv')
config_file = os.path.join(CONFIG_PATH, 'cps.yml')
config = yaml.load(open(config_file, 'r').read())['cps_config']

PATH_TO_DATA_FILE = 'file://' + filename 

spark = SparkSession\
        .builder\
        .appName("spark_example.py")\
        .getOrCreate()

cps_table = spark.read.load(PATH_TO_DATA_FILE,
                         	format='com.databricks.spark.csv',
                         	header='true',
                         	inferSchema='true')
cps_table.createOrReplaceTempView('cps')

projection = ('sex', 'income', 'race')
schema = ektelo_spark.spark_schema(config, projection)
domain = ektelo_spark.spark_domain(config, projection)
config = {field: info for field, info in config.items() if field in projection}
cps_table = spark.sql('SELECT sex, income, race FROM cps WHERE sex = 1')
cps_spark_relation = ektelo_spark.Relation(cps_table, schema, domain, 'cps')
cps_df = cps_spark_relation.to_pandas()
cps_relation = ektelo_data.Relation.from_df(config, cps_df)
