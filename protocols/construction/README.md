# Construction Chronology

These four documentary extracts retain construction evidence from
`scientific_supp_v2`, an archived earlier supplementary-material package, not
its older manuscript, figures or result families. Each JSON records the archive and member SHA-256; the exporter can
reconstruct the projection from a read-only original repository.

- `heldout_selection_documentary_summary.json`: recorded selection rule of the
  confirmation group (`heldout` in code; E2 and E3), 46 frames / 46 participant
  clusters / 92 requests, four zero-overlap counts against the exploratory
  group, and original record/set hashes. The original frame and request identity
  arrays are deliberately absent.
- `hash_crosswalk.json`: original binding hashes and the archived construction
  code hashes. These are not the hashes of the current source export; consult
  `runtime/source_manifest.json` for that separately versioned code.
- `contract_extract.json`: recorded technical-check-run (`canary`) and
  completion requirements and binding fields, without deployment
  account/container paths or output locators.
- `execution_binding_extract.json`: recorded job/worker timing and hash
  consistency, with release entries identified by artifact role, not source path.

All original hashes retain their original meaning. None is presented as the
hash of a redacted document. These records establish recorded local chronology
and content consistency, not independent blinding, externally certified deposit
timing, or a complete account of outcome access. Identity-set hashes do not
allow independent recomputation of overlaps without the original private sets.
The archived construction receipt did not record a process source-file hash;
the archive cannot establish an unrecorded construction-time code digest.

The summary is **not a replacement** for the original hash-bound runtime pack.
The confirmation-group runner and analyzer still require an independently
obtained private execution manifest and the original matching requests,
contracts and receipts.
No execution gates are changed. Original runtime selection bytes are excluded
from this public package, not silently substituted with the summary.

See `docs/preparation.md` for input requirements and the synthetic CPU scope.
