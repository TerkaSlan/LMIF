import pytest
import numpy as np
import pandas as pd
from dlmi.LMI import LMI, Inconsistencies, NodeType
from dlmi.utils import data_X_to_torch
from dlmi.Database import Database


def generated_data():
    np.random.seed(0)
    data = np.random.random((1000, 282))
    return pd.DataFrame(data)


def generated_config():
    return {
        'LMI': {
            'leaf_node_capacity_min': 100,
            'leaf_node_capacity_max': 200,
            'children_min': 5,
            'children_max': 100,
            'violating_nodes': 0.1
        }
    }


@pytest.fixture
def synthetic_data():
    descriptors = generated_data()
    config = generated_config()
    return descriptors, config


def test_LMI_is_created(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1])

    assert len(lmi.inconsistencies[NodeType.LEAF_NODE.value][Inconsistencies.OVERFLOW.value]) == 0
    assert len(lmi.inconsistencies[NodeType.INNER_NODE.value][Inconsistencies.OVERFLOW.value]) == 0
    assert len(lmi.inconsistencies[NodeType.LEAF_NODE.value][Inconsistencies.UNDERFLOW.value]) == 1
    assert len(lmi.nodes) == 1
    assert (0, ) in lmi.nodes
    assert len(lmi.nodes[(0, )].object_ids) == 1


def test_data_is_inserted_incrementally(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:50])
    lmi.insert(data.iloc[50:300])

    assert len(lmi.nodes) == 1
    assert (0, ) in lmi.nodes
    assert lmi.nodes[(0, )].is_overflow() is True
    assert len(lmi.inconsistencies[NodeType.LEAF_NODE.value][Inconsistencies.OVERFLOW.value]) == 1
    assert len(lmi.inconsistencies[NodeType.LEAF_NODE.value][Inconsistencies.UNDERFLOW.value]) == 0
    assert len(lmi.nodes[(0, )].object_ids) == 300


def test_deepen_is_successful_start(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:300])
    lmi.deepen(lmi.nodes[(0,)], 4)

    assert len(lmi.nodes) == 5
    assert len(lmi.nodes[(0, )].children) == 4
    assert (0,) in lmi.inconsistencies[NodeType.INNER_NODE.value][Inconsistencies.UNDERFLOW.value]


def test_LMI_is_retrained_shallow(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:800])
    lmi.deepen(lmi.nodes[(0,)], 4)
    lmi.retrain(lmi.nodes[(0,)], 10)

    assert len(lmi.nodes) == 11
    assert len(lmi.nodes[(0, )].children) == 10


def test_LMI_is_retrained_deeper(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:300])
    lmi.deepen(lmi.nodes[(0,)], 4)
    lmi.retrain(lmi.nodes[(0, )], 5)
    lmi.deepen(lmi.nodes[(0, 1)], 4)
    lmi.retrain(lmi.nodes[(0, )], 10)

    assert len(lmi.nodes) == 11
    assert len(lmi.nodes[(0, )].children) == 10

def test_search(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2)
    lmi.deepen(lmi.nodes[(0, 0)], 2)
    lmi.deepen(lmi.nodes[(0, 1)], 2)
    assert len(lmi.nodes) == 7
    candidate_answer, n_found_objects = lmi.search(data.iloc[0], 100)
    assert len(candidate_answer) == 3
    assert n_found_objects > 100
    assert sum([len(lmi.nodes[leaf_node_pos].objects) for leaf_node_pos in candidate_answer]) == n_found_objects

def test_search(
    synthetic_data
):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2)
    lmi.deepen(lmi.nodes[(0, 0)], 2)
    lmi.deepen(lmi.nodes[(0, 1)], 2)
    assert len(lmi.nodes) == 7
    res = lmi.search(data.iloc[0], 1)
    assert len(res[0]) == 1

def test_LMI_is_retrained_shallow_2(synthetic_data):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2)
    lmi.deepen(lmi.nodes[(0, 0)], 2)
    lmi.deepen(lmi.nodes[(0, 1)], 2)
    lmi.retrain(lmi.nodes[(0, 0)], 4)
    assert len([pos for pos in lmi.nodes.keys() if pos[:2] == (0, 0)]) == 5
    
def test_LMI_is_retrained_deeper_2(synthetic_data):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2)
    lmi.deepen(lmi.nodes[(0, 0)], 2)
    lmi.deepen(lmi.nodes[(0, 1)], 2)
    lmi.deepen(lmi.nodes[(0, 0, 0)], 2)
    lmi.deepen(lmi.nodes[(0, 0, 1)], 2)
    lmi.retrain(lmi.nodes[(0, 0)], 4)
    assert len([pos for pos in lmi.nodes.keys() if pos[:2] == (0, 0)]) == 5

def test_shorten_objects_correctly_classified(synthetic_data):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:500])
    lmi.deepen(lmi.nodes[(0,)], 2)
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 2
    assert all(lmi.nodes[(0, )].nn.predict(data_X_to_torch(lmi.nodes[(0, 0)].objects))) == 0
    lmi.shorten(lmi.nodes[(0, )], [lmi.nodes[(0, 0)]])
    # the other leaf node will inherit all the objects from the shortened one
    assert len(lmi.nodes[(0, 1)].objects) == 500
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 1
    assert all(lmi.nodes[(0, )].nn.predict(data_X_to_torch(lmi.nodes[(0, 1)].objects))) == 1
    
    
def test_shorten_objects_correctly_classified(synthetic_data):
    data, config = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:500])
    lmi.deepen(lmi.nodes[(0,)], 2)
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 2
    assert all(lmi.nodes[(0, )].nn.predict(data_X_to_torch(lmi.nodes[(0, 0)].objects))) == 0
    lmi.shorten(lmi.nodes[(0, )], [lmi.nodes[(0, 0)]])
    # the other leaf node will inherit all the objects from the shortened one
    assert len(lmi.nodes[(0, 1)].objects) == 500
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 1

    
def test_database(synthetic_data):
    data,_ = synthetic_data
    db = Database('config.yml')
    db.insert(data)
    
    assert db.lmi.dump_structure()[db.lmi.dump_structure()['type'] == 'LeafNode']['children'].sum() == data.shape[0]