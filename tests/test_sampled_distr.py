import pytest  # type: ignore

from sampled_distr import UniVarSampledDistr

DEFAULT_OBSERVATIONS = [3, 1, 1, 2]


@pytest.fixture()
def default_distr() -> UniVarSampledDistr:
    distr = UniVarSampledDistr()
    for obs in DEFAULT_OBSERVATIONS:
        distr.observe(obs)

    return distr


def test_init():
    distr = UniVarSampledDistr()
    assert not distr.support_weights
    assert distr.total_weight == 0


def test_observe(default_distr: UniVarSampledDistr):
    assert default_distr.support_weights[1] == 2.0
    assert default_distr.support_weights[2] == 1.0
    assert default_distr.support_weights[3] == 1.0


def test_get_support(default_distr: UniVarSampledDistr):
    assert default_distr.get_support() == [1, 2, 3]


def test_get_probabilities(default_distr: UniVarSampledDistr):
    assert default_distr.get_probabilities() == [1 / 2, 1 / 4, 1 / 4]


def test__call__(default_distr: UniVarSampledDistr):
    assert default_distr.eq(0.0) == 0.0
    assert default_distr.eq(1) == 1 / 2
    assert default_distr.eq(2) == 1 / 4
    assert default_distr.eq(3) == 1 / 4
    assert default_distr.eq(4) == 0.0


def test__lq__(default_distr: UniVarSampledDistr):
    assert default_distr.lt(0) == 0.0
    assert default_distr.lt(1) == 0.0
    assert default_distr.lt(2) == 1 / 2
    assert default_distr.lt(3) == 3 / 4
    assert default_distr.lt(4) == 1.0


def test__le__(default_distr: UniVarSampledDistr):
    assert default_distr.le(0) == 0.0
    assert default_distr.le(1) == 1 / 2
    assert default_distr.le(2) == 3 / 4
    assert default_distr.le(3) == 1.0
    assert default_distr.le(4) == 1.0


def test__ge__(default_distr: UniVarSampledDistr):
    assert default_distr.ge(0) == 1.0
    assert default_distr.ge(1) == 1.0
    assert default_distr.ge(2) == 1 / 2
    assert default_distr.ge(3) == 1 / 4
    assert default_distr.ge(4) == 0.0


def test__gt__(default_distr: UniVarSampledDistr):
    assert default_distr.gt(0) == 1.0
    assert default_distr.gt(1) == 1 / 2
    assert default_distr.gt(2) == 1 / 4
    assert default_distr.gt(3) == 0.0
    assert default_distr.ge(4) == 0.0
