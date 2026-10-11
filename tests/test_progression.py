from copy import deepcopy
from dataclasses import replace
import unittest

from jepa_lab.progression import validate_progression
from jepa_lab.probes import ProbeConfig
from test_compare import sets


def arms():
    first = sets()
    result = {}
    for name, step, fingerprint in (("early", 3, "e"*64), ("later", 30, "f"*64)):
        data = dict(first)
        metadata = first["checkpoint"][0].metadata | dict(completed_steps=step, checkpoint_sha256=fingerprint,
                                                       training_settings=dict(seed=29, batch_size=4))
        data["checkpoint"] = tuple(replace(item, metadata=deepcopy(metadata)) for item in first["checkpoint"])
        result[name] = data
    return result


class ProgressionContractTests(unittest.TestCase):
    def test_ages_and_exact_shared_baselines(self):
        data = arms()
        self.assertEqual([3, 30], [item["completed_steps"] for item in validate_progression(data, (ProbeConfig(steps=2),))])
        for change in ("age", "duplicate", "seed", "pooling", "indices", "baseline", "baseline_values"):
            data = arms()
            train, validation = data["later"]["checkpoint"]
            metadata = deepcopy(train.metadata)
            if change == "age": metadata["completed_steps"] = 3
            if change == "duplicate": metadata["checkpoint_sha256"] = "e"*64
            if change == "seed": metadata["training_settings"]["seed"] = 31
            if change == "pooling": metadata["pooling"] = "different"
            data["later"]["checkpoint"] = replace(train, metadata=metadata), replace(validation, metadata=metadata)
            if change == "indices":
                data["later"] = {name: tuple(replace(item, indices=tuple(reversed(item.indices))) for item in pair)
                                 for name, pair in data["later"].items()}
            if change.startswith("baseline"):
                pair = data["later"]["random"]
                changed = replace(pair[0], artifact_sha256="d"*64) if change == "baseline" else replace(pair[0], features=pair[0].features + 0.1)
                data["later"]["random"] = changed, pair[1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_progression(data, (ProbeConfig(steps=2),))
        for invalid in ({}, {"early": arms()["early"]}, {"../../escape": arms()["early"], "later": arms()["later"]}):
            with self.assertRaises(ValueError): validate_progression(invalid, (ProbeConfig(steps=2),))
