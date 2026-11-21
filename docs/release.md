# Versions and package integrity

Download a versioned package from
https://github.com/0xHAng/NPC-EITN/releases . Version 0.2.3 includes software,
configuration examples, synthetic inputs generated at runtime, aggregate
reference results and documentation. The release assets provide a ZIP and
SHA256SUMS for integrity verification.

```sh
shasum -a 256 NPC_EITN_code_v0.2.3.zip
```

Compare the digest with SHA256SUMS before extracting the archive. After
installation, run python scripts/check_package.py to verify every file against
FILE_MANIFEST.json. GitHub Actions runs the package checks, tests and synthetic
pipeline before creating the version assets.

The ZIP is generated from an explicit file allowlist with fixed file order,
permissions and timestamps. It contains no participant-level data, clinical
model objects, private configuration or cache files. Output directories must
be new and outside the extracted package.

Use the version and exact release URL in citations. See CITATION.md and
CITATION.cff for citation information. GitHub releases are version records;
no archival DOI is assigned. Reuse terms are stated in LICENSE_STATUS.md.
