import pytest
import numpy as np
import pandas as pd
from dlmi.LMI import LMI, Inconsistencies, NodeType
from dlmi.utils import data_X_to_torch, create_dir, remove_dir, Path
from dlmi.Database import Database
from dlmi.search_utils import get_1nn_hit, get_knn_hit, get_objective_knns


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

def generated_info_df():
    return pd.DataFrame([], columns=['op', 'time-taken', 'size', '#-objects'])


@pytest.fixture
def synthetic_data():
    descriptors = generated_data()
    config = generated_config()
    info_df = generated_info_df()
    return descriptors, config, info_df


def test_LMI_is_created(
    synthetic_data
):
    data, config, info_df = synthetic_data
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
    data, config, info_df = synthetic_data
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
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:300])
    lmi.deepen(lmi.nodes[(0,)], 4, info_df)

    assert len(lmi.nodes) == 5
    assert len(lmi.nodes[(0, )].children) == 4
    assert (0,) in lmi.inconsistencies[NodeType.INNER_NODE.value][Inconsistencies.UNDERFLOW.value]


def test_LMI_is_retrained_shallow(
    synthetic_data
):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:800])
    lmi.deepen(lmi.nodes[(0,)], 4, info_df)
    lmi.retrain(lmi.nodes[(0,)], 10, info_df)

    assert len(lmi.nodes) == 11
    assert len(lmi.nodes[(0, )].children) == 10


def test_LMI_is_retrained_deeper(
    synthetic_data
):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:300])
    lmi.deepen(lmi.nodes[(0,)], 4, info_df)
    lmi.retrain(lmi.nodes[(0, )], 5, info_df)
    lmi.deepen(lmi.nodes[(0, 1)], 4, info_df)
    lmi.retrain(lmi.nodes[(0, )], 10, info_df)

    assert len(lmi.nodes) == 11
    assert len(lmi.nodes[(0, )].children) == 10

def test_search(
    synthetic_data
):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 0)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 1)], 2, info_df)
    assert len(lmi.nodes) == 7
    candidate_answer, n_found_objects = lmi.search(data.iloc[0], 100)
    assert len(candidate_answer) == 3
    assert n_found_objects > 100
    assert sum([len(lmi.nodes[leaf_node_pos].objects) for leaf_node_pos in candidate_answer]) == n_found_objects

def test_search(
    synthetic_data
):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 0)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 1)], 2, info_df)
    assert len(lmi.nodes) == 7
    res = lmi.search(data.iloc[0], 1)
    assert len(res[0]) == 1

def test_LMI_is_retrained_shallow_2(synthetic_data):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 0)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 1)], 2, info_df)
    lmi.retrain(lmi.nodes[(0, 0)], 4, info_df)
    assert len([pos for pos in lmi.nodes.keys() if pos[:2] == (0, 0)]) == 5
    
def test_LMI_is_retrained_deeper_2(synthetic_data):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:1000])
    lmi.deepen(lmi.nodes[(0,)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 0)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 1)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 0, 0)], 2, info_df)
    lmi.deepen(lmi.nodes[(0, 0, 1)], 2, info_df)
    lmi.retrain(lmi.nodes[(0, 0)], 4, info_df)
    assert len([pos for pos in lmi.nodes.keys() if pos[:2] == (0, 0)]) == 5

def test_shorten_objects_correctly_classified(synthetic_data):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:500])
    lmi.deepen(lmi.nodes[(0,)], 2, info_df)
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 2
    assert all(lmi.nodes[(0, )].nn.predict(data_X_to_torch(lmi.nodes[(0, 0)].objects))) == 0
    lmi.shorten(lmi.nodes[(0, )], [lmi.nodes[(0, 0)]])
    # the other leaf node will inherit all the objects from the shortened one
    assert len(lmi.nodes[(0, 1)].objects) == 500
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 1
    assert all(lmi.nodes[(0, )].nn.predict(data_X_to_torch(lmi.nodes[(0, 1)].objects))) == 1
    
    
def test_shorten_objects_correctly_classified(synthetic_data):
    data, config, info_df = synthetic_data
    lmi = LMI(**config['LMI'])
    lmi.insert(data.iloc[:500])
    lmi.deepen(lmi.nodes[(0,)], 2, info_df)
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 2
    assert all(lmi.nodes[(0, )].nn.predict(data_X_to_torch(lmi.nodes[(0, 0)].objects))) == 0
    lmi.shorten([lmi.nodes[(0, 0)]])
    # the other leaf node will inherit all the objects from the shortened one
    assert len(lmi.nodes[(0, 1)].objects) == 500
    assert len(lmi.nodes[(0, )].nn.model.output_neurons) == 1

    
def test_database(synthetic_data):
    data, _, info_df = synthetic_data
    dir_path = 'testdir'
    create_dir(dir_path)
    db = Database('config.yml', dir_path)
    info_df = db.insert(data, info_df)
    assert Path(f'{dir_path}/index-0.pkl').exists()
    assert info_df.shape[0] > 0
    Path(f'{dir_path}/index-0.pkl').unlink()
    remove_dir(dir_path)
    assert db.lmi.dump_structure()[db.lmi.dump_structure()['type'] == 'LeafNode']['children'].sum() == data.shape[0]


def test_1nn_hit(synthetic_data):
    data, _, info_df = synthetic_data
    dir_path = 'testdir'
    create_dir(dir_path)
    db = Database('config.yml', dir_path)
    info_df = db.insert(data, info_df)
    query_id = 0
    res = get_1nn_hit(db, data, query_id)
    assert res[0] == query_id
    assert type(res[1]) is tuple
    assert type(res[2]) is tuple
    assert type(res[4]) is int
    assert type(res[5]) is float
    if res[3] == 1:
        assert res[1] == res[2]
    Path(f'{dir_path}/index-0.pkl').unlink()
    remove_dir(dir_path)

def test_knn_hit(synthetic_data):
    data, _, info_df = synthetic_data

    def get_euclidean_distance(object_1, object_2):
        assert object_1.shape == object_2.shape
        return np.linalg.norm(object_1-object_2)

    query_id = 0
    dir_path = 'testdir'
    create_dir(dir_path)
    db = Database('config.yml', dir_path)
    info_df = db.insert(data, info_df)
    nns = get_objective_knns(data.iloc[query_id].values, data)[0]

    query = data.iloc[query_id].values
    dists = []
    for data_point in data.values:
        dists.append(get_euclidean_distance(query, data_point))
    assert np.argmax(dists) == nns[-1]

    n_knn = 2
    res = get_knn_hit(
        db, data, query_id, nns, n_knn, stop_condition_leaf=1
    )
    assert len(res[2]) == 1
    assert type(res[1]) is list
    assert type(res[2]) is list
    Path(f'{dir_path}/index-0.pkl').unlink()
    remove_dir(dir_path)