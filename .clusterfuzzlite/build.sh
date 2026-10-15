#!/bin/bash -eu
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ClusterFuzzLite fuzzer build

project_root="$SRC/scpn-phase-orchestrator"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m unittest discover -s "$project_root/fuzzers" -p test_policy_yaml_runtime.py

for fuzzer in "$project_root"/fuzzers/*_fuzzer.py; do
  fuzzer_basename="$(basename -s .py "$fuzzer")"
  fuzzer_package="${fuzzer_basename}.pkg"
  corpus="$WORK/${fuzzer_basename}-smoke"
  mkdir -p "$corpus"
  cp "$project_root/domainpacks/minimal_domain/policy.yaml" "$corpus/valid.yaml"
  printf '[unterminated' > "$corpus/invalid.yaml"
  python3 "$fuzzer" -runs=4 -max_total_time=15 -rss_limit_mb=1024 "$corpus"

  pyinstaller \
    --distpath "$OUT" \
    --onefile \
    --name "$fuzzer_package" \
    --hidden-import atheris \
    --hidden-import yaml \
    --paths "$project_root/src" \
    "$fuzzer"

  cat > "$OUT/$fuzzer_basename" <<EOF
#!/bin/sh
# LLVMFuzzerTestOneInput marker for fuzzer discovery.
this_dir=\$(dirname "\$0")
"\$this_dir/$fuzzer_package" "\$@"
EOF
  chmod +x "$OUT/$fuzzer_basename"
  "$OUT/$fuzzer_basename" -runs=4 -max_total_time=15 -rss_limit_mb=1024 "$corpus"
done
