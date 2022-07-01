import numpy as np
import pytest
import torch

from xrmocap.data_structure.limbs import Limbs

output_dir = 'test/data/output/data_structure/' +\
    'test_limbs'


def test_new():
    # test only conn
    connections = np.zeros(shape=[10, 2])
    limbs = Limbs(connections=connections)
    assert len(limbs) == 10
    assert limbs.get_connections().shape == (10, 2)
    assert limbs.get_parts() is None
    assert limbs.get_points() is None
    # test conn + parts
    parts = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    limbs = Limbs(connections=connections, parts=parts)
    assert limbs.get_connections().shape == (10, 2)
    assert len(limbs.get_parts()) == 2
    assert limbs.get_points() is None
    # test conn + parts + points
    points = np.zeros(shape=[20, 3])
    limbs = Limbs(connections=connections, parts=parts, points=points)
    assert limbs.get_connections().shape == (10, 2)
    assert len(limbs.get_parts()) == 2
    assert len(limbs.get_points()) == 20
    # test torch
    points = np.zeros(shape=[20, 3])
    limbs = Limbs(
        connections=torch.from_numpy(connections),
        parts=parts,
        points=torch.from_numpy(points))
    assert isinstance(limbs.get_connections(), np.ndarray)
    assert limbs.get_connections().shape == (10, 2)
    assert len(limbs.get_parts()) == 2
    assert len(limbs.get_points()) == 20


def test_set_conections():
    connections = np.zeros(shape=[10, 2])
    limbs = Limbs(connections=connections)
    # set np
    limbs.set_connections(connections + 1)
    assert limbs.get_connections()[0, 0] == 1
    # set torch
    limbs.set_connections(torch.from_numpy(connections + 2))
    assert limbs.get_connections()[0, 0] == 2
    # set list
    with pytest.raises(TypeError):
        limbs.set_connections(connections.tolist())
    # set wrong shape
    with pytest.raises(ValueError):
        limbs.set_connections(connections.reshape(2, 5, 2))
    # set names
    connection_names = [str(x) for x in range(len(connections))]
    limbs.set_connections(connections, connection_names)
    assert len(limbs.connection_names) == len(connection_names)
    # set wrong len, warning
    limbs.set_connections(connections, connection_names[:5])
    assert limbs.connection_names is None


def test_set_parts():
    connections = np.zeros(shape=[10, 2])
    limbs = Limbs(connections=connections)
    parts = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    # set parts
    limbs.set_parts(parts)
    # set wrong part type
    with pytest.raises(TypeError):
        limbs.set_parts(np.asarray(parts))
    # set wrong index type
    with pytest.raises(TypeError):
        limbs.set_parts(np.asarray(parts).astype(np.float64).tolist())
    # set names
    part_names = [str(x) for x in range(len(parts))]
    limbs.set_parts(parts, part_names)
    assert len(limbs.part_names) == len(part_names)
    # set wrong len, warning
    limbs.set_parts(parts, part_names[:1])
    assert limbs.part_names is None


def test_set_points():
    connections = np.zeros(shape=[10, 2])
    limbs = Limbs(connections=connections)
    # test points2d
    points = np.zeros(shape=[20, 2])
    limbs.set_points(points)
    # test points3d
    points = np.zeros(shape=[20, 3])
    limbs.set_points(points)
    # test points3d+conf
    points = np.zeros(shape=[20, 4])
    limbs.set_points(points)
    # test torch
    limbs.set_points(torch.from_numpy(points))
    # test wrong type
    with pytest.raises(TypeError):
        limbs.set_points(points.tolist())
    # test wrong shape
    with pytest.raises(ValueError):
        limbs.set_points(points.reshape(4, 5, 4))


def test_clone():
    connections = np.zeros(shape=[10, 2])
    parts = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    points = np.zeros(shape=[20, 3])
    limbs = Limbs(connections=connections, parts=parts, points=points)
    another_limbs = limbs.clone()
    assert id(another_limbs.connections) != id(limbs.connections)
    assert id(another_limbs.parts) != id(limbs.parts)
    assert id(another_limbs.points) != id(limbs.points)
    assert another_limbs.connection_names is None
    assert another_limbs.part_names is None


def test_get():
    connections = np.zeros(shape=[10, 2])
    parts = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    points = np.zeros(shape=[20, 3])
    limbs = Limbs(connections=connections, parts=parts, points=points)
    assert 10 == len(limbs)
    assert limbs.get_connections().shape == connections.shape
    assert id(parts) == id(limbs.get_parts())
    assert limbs.get_points().shape == points.shape
    assert 'conn_000' in limbs.get_connection_names()
    assert 'part_000' in limbs.get_part_names()
    for i in range(connections.shape[0]):
        connections[i, :] += i
    limbs.set_connections(connections)
    # test by connection names
    conn_dict = limbs.get_connections_by_names()
    assert np.all(conn_dict['conn_001'] == np.ones((2, )))
    # test by parts
    part_dict = limbs.get_connections_in_parts()
    for part_name in ['part_000', 'part_001']:
        conn_list = part_dict[part_name]
        for conn in conn_list:
            assert isinstance(conn, int)
    # test by part names
    limbs.set_parts(parts, ['head', 'foot'])
    part_dict = limbs.get_connections_in_parts()
    for part_name in ['head', 'foot']:
        conn_list = part_dict[part_name]
        for conn in conn_list:
            assert isinstance(conn, int)
