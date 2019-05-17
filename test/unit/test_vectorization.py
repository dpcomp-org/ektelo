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
        filename =  os.path.join(CSV_PATH, 'strok.csv')
        config_file = os.path.join(CONFIG_PATH, 'stroke.yml')
        config = yaml.load(open(config_file, 'r').read())['stroke_2D_config']

        self.schema = data.Schema(config)
        #self.reduced_domain_shape = (10, 2, 7, 2, 2)
        #self.W = workload.RandomRange(None, int(np.prod(self.reduced_domain_shape)), 25)
        self.W = workload.RandomRange(None, int(np.prod(self.schema.shape)), 25)

    def test_grid(self):
        W_log = workload.LogicalWorkload.from_matrix(self.W)
        import ipdb;ipdb.set_trace()
        vec = vectorization.VectorizationDescription(self.W, self.schema)        
