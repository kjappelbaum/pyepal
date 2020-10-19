# -*- coding: utf-8 -*-
"""Testing the Pareto front utilities"""
import numpy as np

from pypal import PALGPy
from pypal.models.gpr import build_model
from pypal.pal.utils import (
    dominance_check,
    dominance_check_jitted,
    dominance_check_jitted_2,
    dominance_check_jitted_3,
    exhaust_loop,
    get_hypervolume,
    get_kmeans_samples,
    get_maxmin_samples,
    is_pareto_efficient,
)


def test_dominance_check():
    """Testing if the dominance check works"""
    assert not dominance_check(np.array([0, 0]), np.array([0, 0]))
    assert dominance_check(np.array([0, 0, 1]), np.array([0, 0, 0]))
    assert dominance_check(np.array([1, 1, 1]), np.array([0, 0, 0]))
    assert dominance_check(np.array([0, 1, 1]), np.array([0, 0, 0]))

    assert not dominance_check(np.array([4.5, 1]), np.array([3.7, 2]))


def test_dominance_check_jitted(binh_korn_points):
    """Test if the jitted dominance check works"""
    _, y = binh_korn_points  # pylint:disable=invalid-name
    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    point = np.array([1.0, 1.0, 1.0])

    assert dominance_check_jitted(point, array)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    point = np.array([-1.0, -1.0, -1.0])

    assert not dominance_check(point, array)

    assert not dominance_check(y[0], y[1:])
    assert not dominance_check(y[2], y[2:])


def test_dominance_check_jitted_2(binh_korn_points):
    """Testing if the dominance check array/point works"""

    _, y = binh_korn_points  # pylint:disable=invalid-name

    assert not dominance_check_jitted_2(y[1:], y[0])
    assert not dominance_check_jitted_2(y[2:], y[2])

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    point = np.array([1.0, 1.0, 1.0])

    assert not dominance_check_jitted_2(array, point)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    point = np.array([0.0, 0.0, 1.0])

    assert dominance_check_jitted_2(array, point)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    point = np.array([0.0, 0.0, 0.8])

    assert dominance_check_jitted_2(array, point)

    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    point = np.array([0.0, 0.0, 1.8])

    assert not dominance_check_jitted_2(array, point)

    pareto_optimal_points = np.array([[0.5, 2], [3, 1], [4, 0.5]])
    new_point = np.array([3.8, 2.1])

    assert not dominance_check_jitted_2(pareto_optimal_points, new_point)

    ups = np.array(
        [
            [1.0, 2.5],
            [3.5, 1.5],
            [4.5, 1.0],
            [1.0, 1.0],
            [2.9, 1.0],
            [2.4, 0.5],
            [1.0, 1.0],
        ]
    )

    assert not dominance_check_jitted_2(ups, np.array([3.7, 2.0]))


def test_dominance_check_jitted_3():
    """Test the domincance check function with workaround around
    masked array"""
    array = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    point = np.array([1.0, 1.0, 1.0])

    assert not dominance_check_jitted_3(array, point, 2)

    array = np.array([[1.0, 2.5], [3.5, 1.5], [4.5, 1.0], [3.9, 2.2]])
    point = np.array([3.7, 2])

    assert not dominance_check_jitted_3(array, point, 3)


def test_is_pareto_efficient():
    """Can we get the indices of the Pareto-efficient points?"""
    array = np.array([[0, 0], [1, 0], [0, 1]])
    assert (is_pareto_efficient(-array) == np.array([False, True, True])).all()


def test_exhaust_loop(binh_korn_points):
    """Testing the exhaust loop"""
    X_binh_korn, y_binh_korn = binh_korn_points  # pylint:disable=invalid-name

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(X_binh_korn, [model_0, model_1], 2, beta_scale=1, restarts=1)
    palinstance.cross_val_points = 0

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    exhaust_loop(palinstance, y_binh_korn)
    assert sum(palinstance.unclassified) == 0
    assert sum(palinstance.discarded) == 0
    assert sum(palinstance.pareto_optimal) == 100
    assert palinstance.number_pareto_optimal_points == 100
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_sampled_points > 0

    # Now, with flipped signs

    y_binh_korn = y_binh_korn * np.array([1, -1])

    sample_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70])
    model_0 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 0)
    model_1 = build_model(X_binh_korn[sample_idx], y_binh_korn[sample_idx], 1)

    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        goals=["max", "min"],
        restarts=1,
    )
    palinstance.cross_val_points = 0

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    exhaust_loop(palinstance, y_binh_korn)
    assert sum(palinstance.unclassified) == 0
    assert sum(palinstance.discarded) == 0
    assert sum(palinstance.pareto_optimal) == 100
    assert palinstance.number_pareto_optimal_points == 100
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_sampled_points > 0

    # Test batch sampling
    palinstance = PALGPy(
        X_binh_korn,
        [model_0, model_1],
        2,
        beta_scale=1,
        goals=["max", "min"],
        restarts=1,
    )
    palinstance.cross_val_points = 0

    palinstance.update_train_set(sample_idx, y_binh_korn[sample_idx])

    exhaust_loop(palinstance, y_binh_korn, batch_size=10)
    assert sum(palinstance.unclassified) == 0
    assert sum(palinstance.discarded) == 0
    assert sum(palinstance.pareto_optimal) == 100
    assert palinstance.number_pareto_optimal_points == 100
    assert palinstance.number_discarded_points == 0
    assert palinstance.number_sampled_points > 0


def test_kmeans_samples(make_random_dataset):
    """test generation of a set of points closest to the k centroids"""
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    solution = get_kmeans_samples(X, 10)
    assert len(solution) == 10


def test_get_maxmin_samples(make_random_dataset):
    """test generation of a set of maximally diverse points"""
    X, _ = make_random_dataset  # pylint:disable=invalid-name
    solution = get_maxmin_samples(X, 10)
    assert len(solution) == 10


def test_get_hypervolume():
    """Test if the hypervolume calculation makes sense"""
    point = np.array([[1, 1]])
    assert get_hypervolume(point, np.array([0, 0])) == 1

    point = np.array([[1, 1, 1]])
    assert get_hypervolume(point, np.array([0, 0, 0])) == 1

    points = np.array([[1, 1, 1], [0.5, 0.5, 0.5]])
    assert get_hypervolume(points, np.array([0, 0, 0])) == 1
