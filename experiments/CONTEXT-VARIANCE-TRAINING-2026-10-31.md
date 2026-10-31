# Context-variance matched training — October 31, 2026

Executed the [complete prospective recipe](CONTEXT-VARIANCE-2026-10-31-PROTOCOL.md) on local CPU, one thread. All six runs finished 2,223 updates, 8,892 training-image draws each, and 29,472 trainable parameters. Both objectives start from scratch. No reserved image or new downstream score was accessed during this training audit.

## Actual image-order and state evidence

| Seed | Unique training images per arm | Shared actual image-order SHA-256 |
|---|---:|---|
| 20261003 | 8072 | `0e5114fff2e4ff0b6a7096c7f41f3b34ce2df8cca6b0a03ccc4424af36ddf260` |
| 20261010 | 8061 | `604a65150857095c653556bfb85a76c4913cde644b3c36bde1997576103bea96` |
| 20261011 | 8112 | `edc4376d80aee3acd086bcbeccda29f3dbe1de4785b6fef31336e25b014e4bc4` |

| Arm / seed | Completed checkpoint SHA-256 |
|---|---|
| prediction-seed20261003 | `2b67069b6ecde7d101a59cdfdf84a578d8a3392e1aefc4670bdf6b2fac4c37fd` |
| prediction-seed20261010 | `b3d779861b940e4f984d7478d74a3f0f4df227c0612271a858039121552a8d1a` |
| prediction-seed20261011 | `9f72d034502e61363a819185fdf1b4b063f8c8cf600571acfe1f2e746393cd56` |
| context_variance-seed20261003 | `dd0a5d93d3257e49908828d95ddcec9f1ad2decd8416ae9734734a034dcb5a0b` |
| context_variance-seed20261010 | `e702b3233fa32adf7e4e13dfae6374ea48b69084153881393b0ee0dcab9fdbf8` |
| context_variance-seed20261011 | `4932a52bbf99deec485e09646abfca6bff8992fb691f294913337bec8c8a4402` |

## Last fixed diagnostic window (steps 2,201–2,223)

| Seed | Prediction-only Smooth L1 | Regularized Smooth L1 | Regularized variance penalty | Regularized visible image-mean std |
|---|---:|---:|---:|---:|
| 20261003 | 0.112245 | 0.046808 | 0.043043 | 1.238886 |
| 20261010 | 0.085251 | 0.075457 | 0.115236 | 1.113776 |
| 20261011 | 0.092772 | 0.102036 | 0.111585 | 1.136234 |

The regularized total loss is prediction loss plus 0.1 × the variance penalty. Total losses across objectives have different definitions. These batch-four, masked-context diagnostics cannot establish useful embeddings, robustness or better classification. The final 23-step window was fixed by the 100-step summary rule; it was not chosen to optimize the picture.

All 228 local tests passed, plus the seeded mask demo and three-step synthetic training smoke. The complete training audit replays finished checkpoints without training or image indexing. Original files and every journal remain local/ignored. The initialization and mask hashes are reconstructed from pinned code; the actual sampled image order is retained in durable journals. They do not authenticate an external producer.

The same 500-row reservation remains unobserved; prior exposure is still 2,746 rows. Next lock all tuning candidates/selected states and conservatively register the reservation before frozen scoring. Official test remains reserved. No paid compute, new downloads, raw data, weights or per-row histories were published.
