import copy
import numpy as np
from operator import add
from numbers import Number
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg.distributed import MatrixEntry
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql.types import FloatType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

type_map = {'integer': IntegerType,
            'float': FloatType,
            'discrete': IntegerType,
            'continuous': FloatType}


def spark_domain(relation_config, projection=None):
    if projection is None:
        projection = relation_config.keys()

    return [max(relation_config[field]['domain'])+1 for field in projection]


def spark_field(name, field_type):
    return StructField(name, field_type)


def spark_schema(relation_config, projection=None):
    if projection is None:
        projection = relation_config.keys()

    return StructType([spark_field(name, type_map[relation_config[name]['type']]()) for name in projection])


def additive_column(df, columns, name):
    """
    Args:
        df: spark dataframe
        columns: a list of columns to add
        name: name of resulting column

    Returns:
        spark dataframe
    """
    return df.withColumn(name, sum([df[field] for field in columns]))


def key_from_tuple(tup, domain):
    mult = 1.0
    key = 0.0
    for i in range(len(tup)):
        key += tup[i]*mult
        mult *= domain[i]

    return int(key)


def tuple_from_key(key, domain):
    tup = []

    for i in range(len(domain)-1):
        dom = domain[i]
        tup.append(int(key % dom))
        key = key - key % dom
        key /= dom

    tup.append(int(key))

    return tuple(tup)


def coordinate_matrix_from_array(arr, sc):
    SQLContext(sc)  # monkey patches toDF

    nonzeros = np.nonzero(arr)

    assert len(nonzeros) == 2, 'arr must have dimension 2'
    
    entry_idxs = list(zip(*list(map(list, nonzeros))))

    parts = max(1, len(entry_idxs) / 500)
    entries = sc.parallelize([MatrixEntry(i, j, arr[i,j]) for i, j in entry_idxs], parts)

    max_i = entries.map(lambda entry: entry.i).max()
    entries = entries.sortBy(lambda entry: entry.i + entry.j * max_i)

    return CoordinateMatrix(entries, arr.shape[0], arr.shape[1])


class Vector():

    def __init__(self, entries, schema, domain, rows=None, cols=None):
        self.schema = schema
        self.domain = domain

        max_i = entries.map(lambda entry: entry.i).max()
        entries = entries.sortBy(lambda entry: entry.i + entry.j * max_i)

        if rows is not None and cols is not None:
            self._data = CoordinateMatrix(entries, rows, cols) 
        else:
            self._data = CoordinateMatrix(entries, np.prod(self.domain), 1) 

    def rows(self):
        return self._data.numRows()

    def cols(self):
        return self._data.numCols()

    def size(self):
        return max(self.rows(), self.cols())

    @staticmethod
    def from_ndarray(arr, schema, domain, sc):
        return Vector.from_array(arr.flatten(order='F'), schema, domain, sc)

    @staticmethod
    def from_array(arr, schema, domain, sc):
        entry_idxs = np.arange(len(arr))[arr >= 1.0]
        parts = max(1, len(entry_idxs) / 500)

        rows = sc.parallelize(entry_idxs, parts)
        cols = sc.parallelize(np.zeros(entry_idxs.shape).astype(np.int), parts)
        vals = sc.parallelize(arr[entry_idxs], parts)
        tups = rows.zip(cols).zip(vals)
        entries = tups.map(lambda tup: MatrixEntry(tup[0][0], tup[0][1], tup[1]))

        return Vector(entries, schema, domain)

    def to_array(self):
        return self._data.toBlockMatrix().toLocalMatrix().values

    def to_ndarray(self):
        domain = self.domain

        return self._data.toBlockMatrix().toLocalMatrix().values.reshape(domain, order='F')

    def indices(self):
        return np.array([entry.i for entry in self._data.entries.collect()])

    def values(self):
        return np.array([entry.value for entry in self._data.entries.collect()])

    def to_relation(self, name):
        domain = self.domain

        tups = self._data.entries.flatMap(lambda entry: int(entry.value)*[tuple_from_key(entry.i, domain)])
        df = tups.toDF(self.schema)

        return Relation(df, self.schema, self.domain, name)

    def transpose(self):
        data = self._data.transpose()
        entries = data.entries

        return Vector(entries, self.schema, self.domain, self.cols(), self.rows())

    def additive_column(self, projection, column_name):
        name = 'vector_' + repr(id(self))

        return self.to_relation(name).additive_column(projection, column_name).to_vector()

    def __add__(self, x):
        if type(x) == Vector:
            blk_data = self._data.toBlockMatrix()
            blk_x = x._data.toBlockMatrix()

            return Vector(blk_data.add(blk_x).toCoordinateMatrix().entries, self.schema, self.domain)
        elif type(x) == CoordinateMatrix:
            entries = self._data.toBlockMatrix().add(x.toBlockMatrix()).toCoordinateMatrix().entries

            return Vector(entries, self.schema, self.domain)
        elif isinstance(x, Number):
            new_entries = self._data.entries.map(lambda entry: MatrixEntry(entry.i, entry.j, entry.value + x))

            return Vector(new_entries, self.schema, self.domain)
        else:
            raise TypeError('cannot add element of type %s' % type(x))

    def __mul__(self, x):
        if type(x) == Vector:
            blk_data = self._data.transpose().toBlockMatrix()
            blk_x = x._data.toBlockMatrix()

            return Vector(blk_data.multiply(blk_x).toCoordinateMatrix().entries, self.schema, self.domain)
        elif type(x) == CoordinateMatrix:
            """
            This method must return a CoordinateMatrix and not a vector because its shape will generally
            not be consistent with the size of the domain, which is a requirement for vectors.
            """
            entries = self._data.toBlockMatrix().multiply(x.toBlockMatrix()).toCoordinateMatrix().entries
            rows = self.rows()
            cols = x.numCols()

            return CoordinateMatrix(entries, rows, cols) 
        elif type(x) == Matrix:
            """
            This method must return a Matrix and not a vector because its shape will generally
            not be consistent with the size of the domain, which is a requirement for vectors.
            """
            entries = self._data.toBlockMatrix().multiply(x._data.toBlockMatrix()).toCoordinateMatrix().entries
            rows = self.rows()
            cols = x.cols()

            return Matrix(entries, rows, cols) 
        elif isinstance(x, Number):
            new_entries = self._data.entries.map(lambda entry: MatrixEntry(entry.i, entry.j, x * entry.value))

            return Vector(new_entries, self.schema, self.domain)
        else:
            raise TypeError('cannot multiply element of type %s' % type(x))


class Matrix():

    def __init__(self, entries, rows=None, cols=None):
        if rows is None:
            rows = entries.map(lambda entry: entry.i).max() + 1
        self._rows = rows

        if cols is None:
            cols = entries.map(lambda entry: entry.j).max() + 1
        self._cols = cols

        entries = entries.sortBy(lambda entry: entry.i + entry.j * self._rows)
        self._data = CoordinateMatrix(entries, self._rows, self._cols) 

    def rows(self):
        return self._rows

    def cols(self):
        return self._cols

    def indices(self):
        return np.array([(entry.i, entry.j) for entry in self._data.entries.collect()])

    def values(self):
        return np.array([entry.value for entry in self._data.entries.collect()])

    def transpose(self):
        data = self._data.transpose()
        entries = data.entries

        return Matrix(entries, self._cols, self._rows)

    @staticmethod
    def from_ndarray(arr, sc):
        mat = coordinate_matrix_from_array(arr, sc)
        return Matrix(mat.entries, rows=mat.numRows(), cols=mat.numCols())

    def to_ndarray(self):
        return self._data.toBlockMatrix().toLocalMatrix().toArray()

    def __add__(self, x):
        if type(x) == Matrix:
            blk_data = self._data.toBlockMatrix()
            blk_x = x._data.toBlockMatrix()

            return Matrix(blk_data.add(blk_x).toCoordinateMatrix().entries, self._rows, self._cols)
        elif isinstance(x, Number):
            new_entries = self._data.entries.map(lambda entry: MatrixEntry(entry.i, entry.j, entry.value + x))

            return Matrix(new_entries, self._rows, self._cols)
        else:
            raise TypeError('cannot add element of type %s' % type(x))

    def __mul__(self, x):
        if type(x) == Vector:
            product_mat = self._data.toBlockMatrix().multiply(x._data.toBlockMatrix()).toCoordinateMatrix()

            return Matrix(product_mat.entries, x.rows(), x.cols())
        elif type(x) == Matrix:
            entries = self._data.toBlockMatrix().multiply(x._data.toBlockMatrix()).toCoordinateMatrix().entries

            return Matrix(entries, self._rows, x.cols()) 
        elif isinstance(x, Number):
            entries = self._data.entries.map(lambda entry: MatrixEntry(entry.i, entry.j, x * entry.value))

            return Matrix(entries, self._rows, self._cols)
        else:
            raise TypeError('cannot multiply element of type %s' % type(x))


class Relation():

    def __init__(self, df, schema, domain, name):
        self.schema = schema
        self.domain = domain
        self.name = name

        self._data = df

    def histogram(self):
        domain = self.domain

        return self._data.rdd.map(lambda x: (x, 1)) \
                             .reduceByKey(add) \
                             .sortBy(lambda key_val: key_from_tuple(key_val[0], domain))

    def additive_column(self, projection, column_name):
        fields = self.schema.fields
        all_field_names = [field.name for field in fields]

        try:
            projection_idxs = [all_field_names.index(column) for column in projection]
        except ValueError:
            raise ValueError('column list provided is invalid')

        field_names = [fields[idx].name for idx in projection_idxs]
        field_types = set([fields[idx].dataType for idx in projection_idxs])

        assert len(field_types) == 1, 'column types must be identical'

        field_type = field_types.pop()
        schema = copy.deepcopy(self.schema)
        schema.add(spark_field(column_name, field_type))

        sub_domain = [self.domain[idx] for idx in projection_idxs]
        domain = self.domain + [sum(sub_domain)]
        df = additive_column(self._data, projection, column_name)

        return Relation(df, schema, domain, self.name)

    def to_pandas(self):
        return self._data.toPandas()

    def to_rdd(self):
        return self._data

    def to_vector(self):
        hist = self.histogram()
        domain = self.domain

        rows = hist.keys().map(lambda tup: key_from_tuple(tup, domain))
        cols = rows.map(lambda x: 0)
        vals = hist.values()
        tups = rows.zip(cols).zip(vals)
        entries = tups.map(lambda tup: MatrixEntry(tup[0][0], tup[0][1], tup[1]))

        return Vector(entries, self.schema, self.domain)
