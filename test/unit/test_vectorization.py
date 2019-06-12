from ektelo import data
from ektelo import vectorization
from ektelo import workload
from ektelo.client.mapper import partition_grid
from ektelo.client.mapper import partition_grid_cells
from ektelo.client.mapper import consolidate_partition_grid
import numpy as np
import os
import unittest
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')


class TestVectorization(unittest.TestCase):

    def setUp(self):
        filename =  os.path.join(CSV_PATH, 'stroke.csv')
        config_file = os.path.join(CONFIG_PATH, 'stroke.yml')
        config = yaml.load(open(config_file, 'r').read())['stroke_2D_config']

        self.schema = data.Schema(config)
        self.R = data.Relation(config).load_csv(filename)

    def test_vectorize_logical(self):
        Dv = vectorization.VectorizationDescription(self.schema)        
        W_log = workload.RandomLogical(Dv, 5)
        vec = self.R.vectorize(Dv)
        W_vec = W_log.vectorize(Dv)
        W_vec @ vec

    def test_partitioning(self):
        histogram = data.Histogram(self.R.df, 
                                   [self.schema.bins(attr) for attr in self.schema.attributes], 
                                   [self.schema.domain(attr) for attr in self.schema.attributes], 
                                   False, 
                                   None).generate()
        Dv = vectorization.VectorizationDescription(self.schema, edges=histogram.edges)        
        W_log = workload.RandomLogical(Dv, 5)
        W_log.vectorize(Dv)

    def test_partition_grid(self):
        histogram = data.Histogram(self.R.df, 
                                   [self.schema.bins(attr) for attr in self.schema.attributes], 
                                   [self.schema.domain(attr) for attr in self.schema.attributes], 
                                   False, 
                                   None).generate()
        domain_shape = [len(e) for e in histogram.edges]
        reduced_domain_shape = (5, 5)
        partition_array = partition_grid(domain_shape, 
                                         partition_grid_cells(domain_shape, reduced_domain_shape))
        edges = consolidate_partition_grid(partition_array, histogram.edges)
        Dv = vectorization.VectorizationDescription(self.schema, edges=edges)        
        W_log = workload.RandomLogical(Dv, 5)
        W_log.vectorize(Dv)
