# Development log

## September 29, 2026

- Started the image experiment foundation with non-overlapping patch extraction and position metadata.
- Added deterministic target-block masks, visible context sampling, and a synthetic CLI inspection demo.
- Added tests for target/context separation, complete grid coverage, input validation, pixel preservation, and seed isolation.
- Model training and custom learned embeddings remain future work, beginning October 1.
- Verification: six unit tests passed; the seeded CLI demo generated 64 patches, 45 context patches, and four target patches. Added GitHub Actions checks for Python 3.11 and 3.13.
