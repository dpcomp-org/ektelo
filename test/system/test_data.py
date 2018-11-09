from collections import OrderedDict
from ektelo.data import Domain
from ektelo.data import Graph
from ektelo.data import Node
from ektelo.data import Relation
from ektelo.data import RelationHelper
import networkx as nx
import pandas as pd
import unittest


class TestData(unittest.TestCase):

    def setUp(self):
        self.relations = {'CPS': RelationHelper('CPS').load(),
                          'STROKE_1D': RelationHelper('STROKE_1D').load(),
                          'STROKE_2D': RelationHelper('STROKE_2D').load(),
                          'ADULT': RelationHelper('ADULT').load()}

    def test_domain_conformance_to_config(self):
        for name, relation in self.relations.items():
            df = relation.df
            config = relation.domain.config
            for field, domain in {field: entry['domain'] for field, entry in config.items()}.items():
                self.assertGreaterEqual(min(df[field]), domain[0], 
                                        'lower domain check failed for field %s in dataset %s' % (field, name))
                self.assertLessEqual(max(df[field]), domain[1],
                                    'upper domain check failed for field %s in dataset %s' % (field, name))
