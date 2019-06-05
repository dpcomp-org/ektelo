from ektelo import data
from ektelo import workload
import numpy as np
import os
import unittest
import yaml

CSV_PATH = os.environ['EKTELO_DATA']
CONFIG_PATH = os.path.join(os.environ['EKTELO_HOME'], 'resources', 'config')


class TestWorkload(unittest.TestCase):

    def setUp(self):
        pass

    def setUp_logical(self):
        filename =  os.path.join(CSV_PATH, 'stroke.csv')
        config_file = os.path.join(CONFIG_PATH, 'stroke.yml')
        config = yaml.load(open(config_file, 'r').read())['stroke_2D_config']

        self.schema = data.Schema(config)
        self.R = data.Relation(config).load_csv(filename)
        self.W_log = workload.RandomLogical(self.schema, 5)

    def test_incomplete_query(self):
        self.setUp_logical()
        test_query = list(self.W_log.predicates)[0]({'age': 10})
