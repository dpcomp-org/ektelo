from collections import OrderedDict
from ektelo import workload
from ektelo import data as ektelo_data
from ektelo import spark as ektelo_spark
import numpy as np
import os
import pytest
from pyspark.sql import SparkSession
from ektelo.plans import spark as standalone
from ektelo.private import transformation
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')


def test_identity(spark_context):
    seed = 10
    eps = 0.1

    spark = SparkSession\
            .builder\
            .appName("test_spark_plans.py")\
            .getOrCreate()

    cps_filename =  os.path.join(CSV_PATH, 'cps.csv')
    cps_config_file = os.path.join(CONFIG_PATH, 'cps.yml')
    cps_config = yaml.load(open(cps_config_file, 'r').read())['cps_config']
    cps_projection = ('income', 'age', 'marital', 'race', 'sex')
    cps_config = {field: info for field, info in cps_config.items() if field in cps_projection}
    PATH_TO_DATA_FILE = 'file://' + cps_filename
    cps_table = spark.read.load(PATH_TO_DATA_FILE,
                                format='com.databricks.spark.csv',
                                header='true',
                                inferSchema='true')
    cps_table.createOrReplaceTempView('cps')
    cps_table = spark.sql('SELECT %s FROM cps' % ','.join(cps_projection))
    cps_schema = ektelo_spark.spark_schema(cps_config, cps_projection)
    cps_domain = (10, 2, 7, 2, 2)
    relation_cps = ektelo_spark.Relation(cps_table, cps_schema, cps_domain, 'cps')
    x_cps = relation_cps.to_vector()

    stroke_filename =  os.path.join(CSV_PATH, 'stroke.csv')
    stroke_config_file = os.path.join(CONFIG_PATH, 'stroke.yml')
    stroke_config = yaml.load(open(stroke_config_file, 'r').read())['stroke_2D_config']
    PATH_TO_DATA_FILE = 'file://' + stroke_filename
    stroke_table = spark.read.load(PATH_TO_DATA_FILE,
                                format='com.databricks.spark.csv',
                                header='true',
                                inferSchema='true')
    stroke_table.createOrReplaceTempView('stroke')
    stroke_schema = ektelo_spark.spark_schema(stroke_config)
    stroke_domain = (64, 64)
    relation_stroke = ektelo_spark.Relation(stroke_table, stroke_schema, stroke_domain, 'stroke')
    x_stroke = relation_stroke.to_vector()

    W_cps = workload.RandomRange(None, x_cps.size(), 25).matrix
    W_cps = ektelo_spark.Matrix.from_ndarray(W_cps, spark_context)
    W_stroke = workload.RandomRange(None, x_stroke.size(), 25).matrix
    W_stroke = ektelo_spark.Matrix.from_ndarray(W_stroke, spark_context)

    x_hat = standalone.Identity().Run(W_cps,
                                      x_cps,
                                      eps,
                                      seed)
    """
    W_cps.dot(x_hat)
    """

    """
    def test_privelet(self):
        x_hat = standalone.Privelet().Run(self.W_cps,
                                          self.x_cps,
                                          self.eps,
                                          self.seed)
        self.W_cps.dot(x_hat)

    def test_h2(self):
        x_hat = standalone.H2().Run(self.W_cps,
                                    self.x_cps,
                                    self.eps,
                                    self.seed)
        self.W_cps.dot(x_hat)

    def test_hb(self):
        domain_shape = (len(self.x_cps),)
        x_hat = standalone.HB(domain_shape).Run(self.W_cps,
                                                self.x_cps,
                                                self.eps,
                                                self.seed)
        self.W_cps.dot(x_hat)

    def test_hb_2D(self):
        x_hat = standalone.HB(self.stroke_domain).Run(self.W_stroke,
                                                      self.x_stroke,
                                                      self.eps,
                                                      self.seed)
        self.W_stroke.dot(x_hat)

    def test_greedy_h(self):
        x_hat = standalone.GreedyH().Run(self.W_cps,
                                         self.x_cps,
                                         self.eps,
                                         self.seed)
        self.W_cps.dot(x_hat)

    def test_uniform(self):
        x_hat = standalone.Uniform().Run(self.W_cps,
                                         self.x_cps,
                                         self.eps,
                                         self.seed)
        self.W_cps.dot(x_hat)

    def test_privBayesLS(self):
        theta = 1
        x_hat = standalone.PrivBayesLS(theta, self.cps_domain).Run(self.W_cps,
                                                                   self.relation_cps,
                                                                   self.eps,
                                                                   self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem(self):
        ratio = 0.5
        rounds = 3
        data_scale = 1e5
        domain_shape = (len(self.x_cps),)
        use_history = True
        x_hat = standalone.Mwem(ratio, 
                                rounds,
                                data_scale,
                                domain_shape,
                                use_history).Run(self.W_cps,
                                                 self.x_cps,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_2D(self):
        ratio = 0.5
        rounds = 3
        data_scale = 1e5
        use_history = True
        x_hat = standalone.Mwem(ratio, 
                                rounds,
                                data_scale,
                                self.stroke_domain,
                                use_history).Run(self.W_stroke,
                                                 self.x_stroke,
                                                 self.eps,
                                                 self.seed)
        self.W_stroke.dot(x_hat)

    def test_ahp(self):
        eta = 0.35
        ratio = 0.85
        x_hat = standalone.Ahp(eta, ratio).Run(self.W_cps,
                                               self.x_cps,
                                               self.eps,
                                               self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa(self):
        ratio = 0.25
        approx = False
        domain_shape = (len(self.x_cps),)
        x_hat = standalone.Dawa(domain_shape, ratio, approx).Run(self.W_cps,
                                                                 self.x_cps,
                                                                 self.eps,
                                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa_2D(self):
        ratio = 0.25
        approx = False
        x_hat = standalone.Dawa(self.stroke_domain, ratio, approx).Run(self.W_stroke,
                                                                       self.x_stroke,
                                                                       self.eps,
                                                                       self.seed)
        self.W_stroke.dot(x_hat)

    def test_quad_tree(self):
        x_hat = standalone.QuadTree().Run(self.W_cps,
                                          self.x_cps,
                                          self.eps,
                                          self.seed)
        self.W_cps.dot(x_hat)

    def test_ugrid(self):
        data_scale = 1e5
        x = self.x_cps.reshape((len(self.x_cps) // 2, 2))
        x_hat = standalone.UGrid(data_scale).Run(self.W_cps,
                                                 x,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_agrid(self):
        data_scale = 1e5
        x = self.x_cps.reshape((len(self.x_cps) // 2, 2))
        x_hat = standalone.AGrid(data_scale).Run(self.W_cps,
                                                 x,
                                                 self.eps,
                                                 self.seed)
        self.W_cps.dot(x_hat)

    def test_dawa_striped(self):
        stripe_dim = 0
        ratio = 0.25
        approx = False
        x_hat = standalone.DawaStriped(ratio, self.cps_domain, stripe_dim, approx).Run(self.W_cps,
                                                                                       self.x_cps,
                                                                                       self.eps,
                                                                                       self.seed)
        self.W_cps.dot(x_hat)

    def test_striped_HB_slow(self):
        stripe_dim = 0
        x_hat = standalone.StripedHB(self.cps_domain, stripe_dim).Run(self.W_cps,
                                                                      self.x_cps,
                                                                      self.eps,
                                                                      self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_variant_b(self):
        ratio = 0.5
        rounds = 3
        x_hat = standalone.MwemVariantB(ratio, rounds, self.x_cps_scale, self.cps_domain, True).Run(self.W_cps,
                                                           self.x_cps,
                                                           self.eps,
                                                           self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_variant_c(self):
        ratio = 0.5
        rounds = 3
        total_noise_scale = 30
        x_hat = standalone.MwemVariantC(ratio, rounds, self.x_cps_scale, self.cps_domain, total_noise_scale).Run(self.W_cps,
                                                           self.x_cps,
                                                           self.eps,
                                                           self.seed)
        self.W_cps.dot(x_hat)

    def test_mwem_variant_d(self):
        ratio = 0.5
        rounds = 3
        total_noise_scale = 30
        x_hat = standalone.MwemVariantD(ratio, rounds, self.x_cps_scale, self.cps_domain, total_noise_scale).Run(self.W_cps,
                                                           self.x_cps,
                                                           self.eps,
                                                           self.seed)
        self.W_cps.dot(x_hat)
    """
