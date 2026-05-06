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
