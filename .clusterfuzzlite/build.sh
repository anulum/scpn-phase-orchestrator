#!/bin/bash -eu
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - ClusterFuzzLite fuzzer build

project_root="$SRC/scpn-phase-orchestrator"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

for fuzzer in "$project_root"/fuzzers/*_fuzzer.py; do
  fuzzer_basename="$(basename -s .py "$fuzzer")"
  fuzzer_package="${fuzzer_basename}.pkg"

  pyinstaller \
    --distpath "$OUT" \
    --onefile \
    --name "$fuzzer_package" \
    --add-data "$project_root/src/scpn_phase_orchestrator/supervisor/policy_rules.py:src/scpn_phase_orchestrator/supervisor" \
    --paths "$project_root/src" \
    "$fuzzer"

  cat > "$OUT/$fuzzer_basename" <<EOF
#!/bin/sh
# LLVMFuzzerTestOneInput marker for fuzzer discovery.
this_dir=\$(dirname "\$0")
"\$this_dir/$fuzzer_package" "\$@"
EOF
  chmod +x "$OUT/$fuzzer_basename"
done
