from ektelo import data
from ektelo import vectorization
from ektelo import workload
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
        self.W_log = workload.RandomLogical(self.schema, 10, 5)

    def test_grid(self):
        Dv = vectorization.VectorizationDescription(self.W_log.vectorize(), self.schema)        
        vec = Dv.vectorize(self.R)
