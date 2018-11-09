from collections import OrderedDict
from ektelo.data import Domain
from ektelo.data import Graph
from ektelo.data import Node
from ektelo.data import Relation
import networkx as nx
import pandas as pd
import unittest


class TestData(unittest.TestCase):

    def setUp(self):
        self.config = {
            'age': {
                'bins': 75,
                'domain': [16,91],
                'type': 'discrete'
            },
            'race': {
                'bins': 3,
                'domain': [1,4],
                'value_map': {
                    'Black': 1,
                    'Other': 2,
                    'White': 3
                },
                'type': 'ordinal'
            }
        }
        self.config = OrderedDict(sorted(self.config.items()))
        self.df = pd.DataFrame(data={'age': [10, 50, 5, 75, 30],
                                     'race': ['Black', 'White', 'Other', 'Black', 'White']}) 
        self.domain = Domain(self.config)
        self.relation = Relation(self.domain, self.df.copy())

    def test_graph(self):
        n1 = Node()
        n2 = Node()
        n3 = Node()

        graph = Graph()
        graph.insert(n1)
        graph.insert(n2)
        graph.insert(n3, after=n1)

        self.assertEqual(sorted(graph.graph.edges()), 
                         sorted([(n1.id, n2.id), (n1.id, n3.id)]))

    def test_relation_meta_data(self):
        self.assertEqual(self.relation.bins, [75, 3])
        self.assertEqual(self.relation.domains, [[16,91], [1,4]])
        self.assertEqual(sorted(self.relation.value_map[1].items()), 
                         [('Black', 1), ('Other', 2), ('White', 3)])

    def test_relation_data(self):
        self.assertListEqual(self.df['age'].values.tolist(), 
                             self.relation.df['age'].values.tolist())
        self.assertListEqual([1,3,2,1,3], 
                             self.relation.df['race'].values.tolist())

    def test_domain_projection(self):
        self.assertTrue(isinstance(self.domain.project('age'), Domain))
        self.assertTrue(isinstance(self.domain.project(['age']), Domain))
        self.assertTrue(isinstance(self.domain.project(('age',)), Domain))

    def test_relation_projection(self):
        self.assertTrue(isinstance(self.relation.project('age'), Relation))
        self.assertTrue(isinstance(self.relation.project(['age']), Relation))
        self.assertTrue(isinstance(self.relation.project(('age',)), Relation))
