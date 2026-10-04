from collections import Counter
from pathlib import Path
import json
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_lab.cifar10 import CIFAR10Binary, RECORD_BYTES, RECORDS_PER_BATCH, make_manifest, patchify_rgb, stratified_split
from jepa_lab.cifar10_smoke import train_cifar10_smoke
from jepa_lab.masking import block_mask
from jepa_lab.model import JEPA, ModelConfig


class CIFAR10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rgb_patch_order_scaling_and_model_integration(self):
        image = torch.arange(3*4*4, dtype=torch.uint8).reshape(1, 3, 4, 4)
        patches = patchify_rgb(image, patch_size=2)
        self.assertEqual(patches.shape, (1, 4, 12))
        first_pixels = torch.tensor([0, 1, 4, 5, 16, 17, 20, 21, 32, 33, 36, 37], dtype=torch.float32)
        self.assertTrue(torch.equal(patches[0, 0], first_pixels/127.5-1))
        black_white = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        black_white[1].fill_(255)
        prepared = patchify_rgb(black_white)
        self.assertEqual(prepared.shape, (2, 64, 48))
        self.assertTrue((prepared[0] == -1).all())
        self.assertTrue((prepared[1] == 1).all())
        torch.manual_seed(29)
        model = JEPA(ModelConfig(patch_dim=48, embedding_dim=16, encoder_depth=1))
        masks = [block_mask(8, 8, target_rows=2, target_cols=2, seed=29+i) for i in range(2)]
        loss = model.loss(prepared, [m.context for m in masks], [m.target for m in masks])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.context_encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.target_encoder.parameters()))

    def test_invalid_rgb_inputs_are_rejected(self):
        for image in (torch.zeros(1,3,4,4), torch.zeros(3,4,4,dtype=torch.uint8),
                      torch.zeros(0,3,4,4,dtype=torch.uint8), torch.zeros(1,1,4,4,dtype=torch.uint8),
                      torch.zeros(1,3,5,4,dtype=torch.uint8), torch.zeros(1,3,0,4,dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                patchify_rgb(image, patch_size=2)
        for size in (0, True, 2.0):
            with self.assertRaises(ValueError):
                patchify_rgb(torch.zeros(1,3,4,4,dtype=torch.uint8), patch_size=size)

    def test_stable_stratified_partitions_cover_each_image_without_rng_mutation(self):
        labels = [label for label in range(10) for _ in range(12)]
        random.seed(41)
        original_rng = random.getstate()
        split = stratified_split(labels, validation_per_class=2, seed=73)
        self.assertEqual(random.getstate(), original_rng)
        self.assertEqual(split, stratified_split(labels, validation_per_class=2, seed=73))
        self.assertNotEqual(split, stratified_split(labels, validation_per_class=2, seed=74))
        self.assertTrue(set(split["train"]).isdisjoint(split["validation"]))
        self.assertEqual(set(split["train"]+split["validation"]), set(range(len(labels))))
        self.assertEqual(Counter(labels[index] for index in split["validation"]), Counter({i:2 for i in range(10)}))
        for bad_labels, count in (([],1), ([0],1), ([10,10],1), ([True,False],1), ([0,0],0)):
            with self.assertRaises(ValueError):
                stratified_split(bad_labels, validation_per_class=count)

    def test_binary_loader_preserves_channels_and_rejects_corrupt_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/"test_batch.bin"
            data = bytearray(RECORD_BYTES*RECORDS_PER_BATCH)
            data[0] = 7
            data[1:1025] = bytes([23])*1024
            data[1025:2049] = bytes([45])*1024
            data[2049:3073] = bytes([67])*1024
            path.write_bytes(data)
            dataset = CIFAR10Binary(root, train=False)
            image, label = dataset[0]
            self.assertEqual(label, 7)
            self.assertEqual(image.shape, (3,32,32))
            self.assertEqual(image[:,0,0].tolist(), [23,45,67])
            for bad_index in (-1, 10000, 1.0, True):
                with self.assertRaises(IndexError):
                    dataset[bad_index]
            data[0] = 255
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                dataset[0]
            with self.assertRaises(ValueError):
                CIFAR10Binary(root, train=False)
            path.write_bytes(b"truncated")
            with self.assertRaises(ValueError):
                CIFAR10Binary(root, train=False)

    def test_full_manifest_keeps_official_test_separate_and_fingerprints_provenance(self):
        class Metadata:
            def __init__(self, train):
                self.train = train
                self.labels = [index%10 for index in range(50000 if train else 10000)]
                self.batch_sha256 = {"train" if train else "test": "synthetic-checksum"}
            def __len__(self):
                return len(self.labels)
        train, test = Metadata(True), Metadata(False)
        manifest = make_manifest(train, test)
        self.assertEqual({name:len(split["indices"]) for name,split in manifest["partitions"].items()},
                         dict(train=45000, validation=5000, test=10000))
        self.assertEqual(manifest["partitions"]["test"]["official_split"], "test")
        self.assertTrue(set(manifest["partitions"]["train"]["indices"]).isdisjoint(
            manifest["partitions"]["validation"]["indices"]))
        self.assertEqual(make_manifest(train,test),manifest)
        test.batch_sha256["test"] = "changed-checksum"
        self.assertNotEqual(make_manifest(train,test)["manifest_sha256"],manifest["manifest_sha256"])
        with self.assertRaises(ValueError):
            make_manifest(test, train)

    def test_public_smoke_is_reproducible_and_never_samples_validation_or_test(self):
        accessed = []
        class Fixture:
            def __init__(self, root, *, train=True):
                self.train = train
                self.labels = [index%10 for index in range(50000 if train else 10000)]
                self.batch_sha256 = {"train" if train else "test": "synthetic-checksum"}
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, index):
                accessed.append((self.train,index))
                return torch.full((3,32,32), index%256, dtype=torch.uint8), self.labels[index]
        manifest = make_manifest(Fixture("unused"), Fixture("unused", train=False))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"manifest.json"
            path.write_text(json.dumps(manifest))
            with patch("jepa_lab.cifar10_smoke.CIFAR10Binary", Fixture):
                first = train_cifar10_smoke(folder, path, steps=1, batch_size=2)
                second = train_cifar10_smoke(folder, path, steps=1, batch_size=2)
                self.assertEqual(first, second)
                self.assertTrue(all(train and index in manifest["partitions"]["train"]["indices"]
                                    for train,index in accessed))
                self.assertTrue(all(index not in manifest["partitions"]["validation"]["indices"]
                                    for _,index in accessed))
                manifest["partitions"]["train"]["indices"][0] = manifest["partitions"]["validation"]["indices"][0]
                path.write_text(json.dumps(manifest))
                accessed.clear()
                with self.assertRaises(ValueError):
                    train_cifar10_smoke(folder,path)
                self.assertEqual(accessed, [])

    def test_public_smoke_rejects_unbounded_settings_before_reading_data(self):
        for settings in (dict(steps=0), dict(steps=51), dict(steps=True),
                         dict(batch_size=0), dict(batch_size=33), dict(learning_rate=float("nan"))):
            with self.assertRaises(ValueError):
                train_cifar10_smoke("absent", "absent", **settings)


if __name__ == "__main__":
    unittest.main()
