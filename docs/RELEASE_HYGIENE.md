# Release Hygiene

This project pins GitHub Actions by full commit SHA in release workflows. Before
tagging a release, validate those pins so GitHub Actions does not fail during
job setup after the tag has already been pushed.

Run:

```bash
python tools/check_github_action_refs.py .github/workflows/publish.yml
python tools/check_version_sync.py
GITHUB_REF_NAME=vX.Y.Z python tools/check_release_tag_version.py
```

The action-reference guard checks every remote `owner/repo@ref` action in the
workflow, requires full 40-character commit SHAs, and verifies each ref resolves
through the GitHub API. Local actions and `docker://` actions are ignored.

If a tag-triggered publish run fails before uploading, delete the failed tag,
bump the release metadata, fix the workflow on a pull request, and tag the new
version only after CI and the local release guards pass.

If PyPI upload succeeds but a later release job fails, do not reuse the same
version. PyPI files are immutable, so the repair must land on `main` with a new
patch version and a new tag.

For Rust wheel builds in `publish.yml`, Linux manylinux jobs must pass an
explicit CPython interpreter path such as
`/opt/python/cp312-cp312/bin/python` to maturin. The manylinux container PATH
does not guarantee a usable `python3` selector for every target.

For the container job, refresh pinned base-image digests before tagging. A stale
`tag@sha256` pair fails before security scanning because Docker cannot resolve
the source metadata. The container FFI builder should use the same CPython minor
as the runtime image and install the stable Rust toolchain used by CI, not the
workspace MSRV, because locked transitive crates may require newer Cargo
manifest support and a mismatched CPython wheel will not import at runtime.

## Fuzzing Findings

ClusterFuzzLite failures are treated as release blockers when the crash is
reproducible. For policy YAML fuzzing, malformed YAML parser failures must be
contained at the loader boundary and reported as `ValueError` parse errors,
rather than escaping as uncaught parser exceptions. Add a regression test with
the minimized payload before rerunning or merging the release PR.
