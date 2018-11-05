from collections import OrderedDict
from ektelo.data import Graph
from ektelo.data import Node
from ektelo.data import Relation
import networkx as nx
import pandas as pd
import unittest


class TestData(unittest.TestCase):

    def setUp(self):
        pass

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
        config = {
            'age': {
                'bins': 75,
                'domain': [16,91]
            },
            'race': {
                'bins': 3,
                'domain': [1,4],
                'value_map': {
                    'Black': 1,
                    'Other': 2,
                    'White': 3
                }
            }
        }
        config = OrderedDict(sorted(config.items()))
        df = pd.DataFrame(data={'age': [10, 50, 5, 75, 30],
                               'race': ['Black', 'White', 'Other', 'Black', 'White']}) 

        relation = Relation(config, df)
        self.assertEqual(relation.bins, [75, 3])
        self.assertEqual(relation.domains, [[16,91], [1,4]])
        self.assertEqual(sorted(relation.value_map[1].items()), 
                         [('Black', 1), ('Other', 2), ('White', 3)])
