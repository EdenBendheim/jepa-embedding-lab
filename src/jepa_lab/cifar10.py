"""Official CIFAR-10 binary records, RGB patches, and fixed experiment splits."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


SOURCE = "https://www.cs.toronto.edu/~kriz/cifar.html"
ARCHIVE_URL = "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz"
ARCHIVE_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
RECORD_BYTES = 3073
RECORDS_PER_BATCH = 10000
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def stratified_split(labels, *, validation_per_class: int = 500, seed: int = 20261003) -> dict:
    """Stable within-class SHA-256 ordering; no global random state or Python shuffle dependency."""
    if type(validation_per_class) is not int or validation_per_class < 1 or type(seed) is not int:
        raise ValueError("Validation count must be positive and seed must be an integer")
    grouped = {}
    for index, label in enumerate(labels):
        if type(label) is not int or not 0 <= label < len(CLASSES):
            raise ValueError("CIFAR-10 labels must be integers from 0 to 9")
        grouped.setdefault(label, []).append(index)
    if not grouped:
        raise ValueError("Labels cannot be empty")
    train, validation = [], []
    for label, indices in sorted(grouped.items()):
        if len(indices) <= validation_per_class:
            raise ValueError("Every class must retain both train and validation samples")
        ordered = sorted(indices, key=lambda index: (
            hashlib.sha256(f"cifar10-v1:{seed}:{index}".encode("ascii")).digest(), index))
        validation.extend(ordered[:validation_per_class])
        train.extend(ordered[validation_per_class:])
    return dict(train=sorted(train), validation=sorted(validation))


def patchify_rgb(images, *, patch_size: int = 4):
    """Convert uint8 NCHW RGB images to row-major patches with channel-major pixels.

    Use the fixed [0,255] -> [-1,1] transform; never fit normalization on held-out data.
    """
    import torch
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    if not isinstance(images, torch.Tensor) or images.dtype != torch.uint8:
        raise ValueError("RGB images must be a uint8 torch tensor")
    if images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3:
        raise ValueError("RGB images must have nonempty shape [batch, 3, height, width]")
    height, width = images.shape[-2:]
    if not height or not width or height % patch_size or width % patch_size:
        raise ValueError("Image dimensions must be positive and divisible by patch_size")
    values = images.float().div(127.5).sub(1)
    values = values.reshape(images.shape[0], 3, height//patch_size, patch_size,
                            width//patch_size, patch_size)
    return values.permute(0, 2, 4, 1, 3, 5).reshape(
        images.shape[0], (height//patch_size)*(width//patch_size), 3*patch_size*patch_size)


class CIFAR10Binary:
    """Read individual official binary records without pickle, network, or eager image loading."""

    def __init__(self, root: str | Path, *, train: bool = True):
        self.root = Path(root)
        self.train = train
        self.names = [f"data_batch_{index}.bin" for index in range(1, 6)] if train else ["test_batch.bin"]
        self.labels = []
        self.batch_sha256 = {}
        for name in self.names:
            path = self.root/name
            if path.stat().st_size != RECORDS_PER_BATCH*RECORD_BYTES:
                raise ValueError(f"Unexpected CIFAR-10 batch size: {name}")
            data = path.read_bytes()
            labels = list(data[::RECORD_BYTES])
            if any(label >= len(CLASSES) for label in labels):
                raise ValueError(f"Invalid CIFAR-10 class label: {name}")
            self.labels.extend(labels)
            self.batch_sha256[name] = hashlib.sha256(data).hexdigest()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        import torch
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("CIFAR-10 index outside the official split")
        batch, offset = divmod(index, RECORDS_PER_BATCH)
        with (self.root/self.names[batch]).open("rb") as stream:
            stream.seek(offset*RECORD_BYTES)
            record = stream.read(RECORD_BYTES)
        if len(record) != RECORD_BYTES or record[0] != self.labels[index]:
            raise ValueError("CIFAR-10 batch changed since its initial inspection")
        image = torch.frombuffer(bytearray(record[1:]), dtype=torch.uint8).reshape(3, 32, 32)
        return image, record[0]


def make_manifest(train: CIFAR10Binary, test: CIFAR10Binary, *, seed: int = 20261003) -> dict:
    if not train.train or test.train or len(train) != 50000 or len(test) != 10000:
        raise ValueError("Manifest requires official training and test partitions")
    if Counter(train.labels) != Counter({label: 5000 for label in range(10)}) or (
        Counter(test.labels) != Counter({label: 1000 for label in range(10)})):
        raise ValueError("Official CIFAR-10 class counts do not match")
    splits = stratified_split(train.labels, seed=seed)
    manifest = dict(
        version=1, dataset="CIFAR-10 binary", source=SOURCE,
        archive=dict(url=ARCHIVE_URL, md5=ARCHIVE_MD5),
        seed=seed, algorithm="cifar10-v1 SHA-256 ordering within each class",
        classes=list(CLASSES), batches=train.batch_sha256 | test.batch_sha256,
        input=dict(layout="NCHW RGB uint8", normalization="x / 127.5 - 1", patch_size=4,
                   grid=[8, 8], patch_dim=48),
        partitions=dict(train=dict(official_split="train", indices=splits["train"]),
                        validation=dict(official_split="train", indices=splits["validation"]),
                        test=dict(official_split="test", indices=list(range(len(test))))))
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest | {"manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description="Inspect local official CIFAR-10 binaries and record fixed splits")
    parser.add_argument("root", type=Path, help="Extracted cifar-10-batches-bin directory")
    parser.add_argument("--manifest", type=Path, required=True, help="Output JSON; use ignored runs/")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    try:
        manifest = make_manifest(CIFAR10Binary(args.root), CIFAR10Binary(args.root, train=False), seed=args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(manifest_sha256=manifest["manifest_sha256"],
                          counts={name: len(partition["indices"]) for name, partition in manifest["partitions"].items()},
                          input=manifest["input"]), indent=2))


if __name__ == "__main__":
    main()
