# MIF merge-compression fixture

`mif_merge_compression_observation_v1.json` is the immutable public output of
SCPN-MIF-CORE commit `a1a83eb04becebeb0e0c8d05a88b5a90a2cef5a4`. The embedded
`source_revision` intentionally identifies the producer code revision used by
the producer's model-state fixture (`f60dbae4b2ea3344ac0cb086a3b7d248d65cf92f`).

- Length: 2,475 bytes
- SHA-256: `c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca`
- Schema: `scpn-mif-core.merge-compression-observation.v1`
- Authority: review-only, non-actionable simulation evidence

Tests consume these exact bytes through the SPO public adapter. They do not
import or execute the sibling MIF source tree.
