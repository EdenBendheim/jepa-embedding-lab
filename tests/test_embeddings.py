from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import make_manifest, patchify_rgb
from jepa_lab.embeddings import extract_cifar10, image_embeddings, pixel_features, random_encoder
from jepa_lab.model import JEPA, ModelConfig


class Fixture:
    accessed = []

    def __init__(self, root, *, train=True):
        self.train = train
        self.labels = [index%10 for index in range(50000 if train else 10000)]
        self.batch_sha256 = {"train" if train else "test": "synthetic-checksum"}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        self.accessed.append((self.train, index))
        return ((torch.arange(3*32*32)+index)%256).to(torch.uint8).reshape(3,32,32), self.labels[index]


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.manifest = make_manifest(Fixture(self.root), Fixture(self.root, train=False))
        self.path = self.root/"manifest.json"
        self.path.write_text(json.dumps(self.manifest))
        self.config = ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1)
        self.model = random_encoder(self.config, 29)
        self.checkpoint = self.root/"encoder.pt"
        self.state = dict(format_version=1, dataset="CIFAR-10", manifest_sha256=self.manifest["manifest_sha256"],
                          runtime=dict(torch=str(torch.__version__), cpu_threads=1), step=3,
                          config=asdict(self.config), model=self.model.state_dict(), settings=dict(seed=29))
        torch.save(self.state, self.checkpoint)
        self.patcher = patch("jepa_lab.embeddings.CIFAR10Binary", Fixture)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        Fixture.accessed.clear()

    def extract(self, **settings):
        options = dict(limit=4, batch_size=3, checkpoint=self.checkpoint)
        options.update(settings)
        return extract_cifar10(self.root, self.path, **options)

    def test_full_image_pooling_uses_all_positions_without_gradients_and_restores_mode(self):
        images = torch.stack([Fixture(self.root)[index][0] for index in (0, 1)])
        model = JEPA(self.config).train()
        before = {name:value.clone() for name,value in model.state_dict().items()}
        features = image_embeddings(model, images)
        self.assertTrue(model.training)
        self.assertFalse(model.target_encoder.training)
        self.assertFalse(features.requires_grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        model.eval()
        with torch.no_grad():
            expected = model.context_encoder(patchify_rgb(images), torch.arange(64).expand(2,-1)).mean(1)
        self.assertTrue(torch.equal(features, expected))
        self.assertTrue(all(torch.equal(value, before[name]) for name,value in model.state_dict().items()))
        target = image_embeddings(model, images, encoder="target")
        self.assertEqual((2,16), tuple(target.shape))
        model.train()
        with patch.object(model.context_encoder, "forward", side_effect=RuntimeError("simulated encoder failure")), \
                self.assertRaises(RuntimeError):
            image_embeddings(model, images)
        self.assertTrue(model.training)

    def test_raw_pixel_features_preserve_nchw_order_and_fixed_scaling(self):
        images = torch.zeros(2,3,32,32, dtype=torch.uint8)
        images[1].fill_(255)
        images[0,1,0,0] = 255
        values = pixel_features(images)
        self.assertEqual((2,3072), tuple(values.shape))
        self.assertEqual(1.0, float(values[0,1024]))
        self.assertEqual(-1.0, float(values[0,0]))
        self.assertTrue((values[1] == 1).all())
        self.assertFalse(values.requires_grad)
        with self.assertRaises(ValueError):
            pixel_features(torch.zeros(1,3,16,16, dtype=torch.uint8))

    def test_seeded_random_encoder_does_not_change_caller_rng(self):
        state = torch.get_rng_state().clone()
        first, second, different = (random_encoder(self.config, seed) for seed in (29,29,30))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        image = Fixture(self.root)[0][0].unsqueeze(0)
        self.assertTrue(torch.equal(image_embeddings(first,image), image_embeddings(second,image)))
        self.assertFalse(torch.equal(image_embeddings(first,image), image_embeddings(different,image)))
        self.assertTrue(all(not parameter.requires_grad for parameter in first.parameters()))

    def test_checkpoint_features_are_reproducible_and_artifact_binds_exact_partition_and_weights(self):
        output = self.root/"features.pt"
        state = torch.get_rng_state().clone()
        first = self.extract(output=output)
        second = self.extract()
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        artifact = torch.load(output, weights_only=True)
        self.assertEqual(self.manifest["partitions"]["train"]["indices"][:4], artifact["indices"])
        self.assertEqual([index%10 for index in artifact["indices"]], artifact["labels"].tolist())
        self.assertEqual((4,16), tuple(artifact["features"].shape))
        self.assertFalse(artifact["features"].requires_grad)
        self.assertEqual(self.manifest["manifest_sha256"], artifact["manifest_sha256"])
        self.assertEqual(3, artifact["feature_metadata"]["completed_steps"])
        self.assertEqual(64, len(artifact["feature_metadata"]["checkpoint_sha256"]))
        self.assertTrue(all(is_train and index in artifact["indices"] for is_train,index in Fixture.accessed))

    def test_baselines_and_validation_test_partitions_keep_namespace_and_example_identity(self):
        outputs = []
        for representation in ("random", "pixels"):
            output = self.root/f"{representation}.pt"
            self.extract(representation=representation, checkpoint=None, config=self.config, output=output)
            outputs.append(torch.load(output, weights_only=True))
        self.assertEqual(outputs[0]["indices"], outputs[1]["indices"])
        self.assertEqual(outputs[0]["labels"].tolist(), outputs[1]["labels"].tolist())
        self.assertEqual((4,16), tuple(outputs[0]["features"].shape))
        self.assertEqual((4,3072), tuple(outputs[1]["features"].shape))
        for partition, official in (("validation", True), ("test", False)):
            Fixture.accessed.clear()
            report = self.extract(partition=partition)
            expected = self.manifest["partitions"][partition]["indices"][:4]
            self.assertEqual([(official,index) for index in expected], Fixture.accessed)
            self.assertEqual("train" if official else "test", report["official_split"])

    def test_mismatched_checkpoint_and_edited_manifest_fail_before_sampling_or_output_writes(self):
        output = self.root/"features.pt"
        output.write_bytes(b"existing artifact")
        for changes in (dict(manifest_sha256="wrong"), dict(runtime=dict(torch="wrong-version")), dict(step=True)):
            torch.save(self.state | changes, self.checkpoint)
            Fixture.accessed.clear()
            with self.assertRaises(ValueError):
                self.extract(output=output)
            self.assertEqual([], Fixture.accessed)
            self.assertEqual(b"existing artifact", output.read_bytes())
        torch.save(self.state, self.checkpoint)
        self.manifest["partitions"]["validation"]["indices"].reverse()
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Manifest"):
            self.extract(output=output)
        self.assertEqual([], Fixture.accessed)

    def test_invalid_bounds_encoder_and_nonfinite_features_are_rejected(self):
        for changes in (dict(limit=0), dict(limit=1001), dict(batch_size=True), dict(encoder="predictor"),
                        dict(representation="random"), dict(partition="other")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.extract(**changes)
        self.assertEqual([], Fixture.accessed)
        images = torch.zeros(1,3,32,32, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            image_embeddings(self.model, torch.zeros(1,3,16,16,dtype=torch.uint8))
        with torch.no_grad():
            self.model.context_encoder.projection.weight.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            image_embeddings(self.model, images)


if __name__ == "__main__":
    unittest.main()
