"""datastore(): the sampler owns population, version and cardinality."""

import ifed
import pytest

from opaque.federated import MinSepSampler, datastore
from opaque.federated import population as make_population


@pytest.fixture
def sampler():
    return MinSepSampler(make_population("/hive", version="1.*"), batch_size=8, bands=4)


def test_takes_population_and_cardinality_from_sampler(sampler):
    store = datastore(sampler)
    assert isinstance(store, ifed.FederatedDatastore)
    assert store.population == "/hive"
    assert store.version == "1.*"
    assert store.cardinality == 8


def test_passes_requirements_through(sampler):
    store = datastore(sampler, server="stgn", gpu=False, cpu_count=4)
    assert store.server == "stgn"
    assert store.gpu is False
    assert store.cpu_count == 4


@pytest.mark.parametrize("owned", ["population", "version", "cardinality"])
def test_rejects_sampler_owned_keywords(sampler, owned):
    with pytest.raises(TypeError, match=owned):
        datastore(sampler, **{owned: 3 if owned == "cardinality" else "/other"})
