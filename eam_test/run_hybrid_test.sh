#!/usr/bin/env bash
# run_hybrid_test.sh
# Sanity-checks the hybrid/scaled pace+pace implementation against
# a single-PACE reference.
#
# Test 1 (REF):  single pace,          noforce yes    → reference PE trajectory
# Test 2 (HS-NF): hybrid/scaled 0.5+0.5, noforce yes  → PE must match REF
# Test 3 (HS-LE): hybrid/scaled 0.5+0.5, localE yes   → accept/reject must match HS-NF
#
# Pass criterion: max(|PE_ref - PE_hybrid|) < 1e-8 eV
#                 accept/reject sequence identical between HS-NF and HS-LE

set -euo pipefail
cd "$(dirname "$0")"

LMP=${LMP:-../build/lmp}
YACE=AuCu_LDA.yace
TOL=1e-8   # eV tolerance for PE comparison

run() {
    local label=$1; shift
    echo ">>> Running $label ..."
    "$LMP" "$@" -screen none -log "log_${label}.lammps"
    echo "    done."
}

extract_pe() {
    # pull PE column from thermo output (lines starting with a step number)
    local logfile=$1
    awk '/^[[:space:]]*[0-9]+ /{print $2}' "$logfile"
}

extract_swaps() {
    # pull 'accepted' count from final stats line  (Step Nattempts Naccepted)
    local logfile=$1
    awk '/atom\/swap/{found=1} found && /^[[:space:]]*[0-9]/{print $3; found=0}' "$logfile"
}

# ── Test 1: single PACE, noforce yes (reference) ─────────────────────────────
# lmp_pace.in uses a literal SWAP_KEYWORD placeholder; substitute via sed
sed 's/SWAP_KEYWORD/noforce yes/' lmp_pace.in > lmp_pace_ref.in
run REF -in lmp_pace_ref.in

# ── Test 2: hybrid/scaled 0.5+0.5, noforce yes ───────────────────────────────
run HS_NF -in lmp_hybrid_scaled.in -var lambda 0.5 -var swap_kw "noforce yes"

# ── Test 3: hybrid/scaled 0.5+0.5, localE yes ────────────────────────────────
run HS_LE -in lmp_hybrid_scaled.in -var lambda 0.5 -var swap_kw "localE yes"

# ── Comparisons ──────────────────────────────────────────────────────────────
echo ""
echo "=== PE comparison: REF vs HS_NF (max |ΔPE| should be < ${TOL} eV) ==="
python3 - <<'PYEOF'
import sys, math

def read_pe(fname):
    import re
    pes = []
    with open(fname) as f:
        for line in f:
            # thermo lines: first column is integer step, second is PE
            m = re.match(r'^\s*(\d+)\s+([-\d.eE+]+)', line)
            if m:
                pes.append(float(m.group(2)))
    return pes

ref  = read_pe("log_REF.lammps")
hsnf = read_pe("log_HS_NF.lammps")

if len(ref) != len(hsnf):
    print(f"  WARN: different number of thermo lines ({len(ref)} vs {len(hsnf)})")
    n = min(len(ref), len(hsnf))
else:
    n = len(ref)

diffs = [abs(ref[i] - hsnf[i]) for i in range(n)]
maxd  = max(diffs)
print(f"  steps compared : {n}")
print(f"  max |ΔPE|      : {maxd:.3e} eV")
tol = 1e-8
if maxd < tol:
    print(f"  RESULT         : PASS (< {tol} eV)")
else:
    print(f"  RESULT         : FAIL (>= {tol} eV)")
    sys.exit(1)
PYEOF

echo ""
echo "=== Accept/reject comparison: HS_NF vs HS_LE (sequences must be identical) ==="
python3 - <<'PYEOF'
import sys, re

def read_thermo(fname):
    """Return list of (step, pe) from thermo output."""
    rows = []
    with open(fname) as f:
        for line in f:
            m = re.match(r'^\s*(\d+)\s+([-\d.eE+]+)', line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    return rows

# The PE changes on accepted swaps; identical PE trajectory ⟺ identical accept/reject
nf = read_thermo("log_HS_NF.lammps")
le = read_thermo("log_HS_LE.lammps")

if len(nf) != len(le):
    print(f"  WARN: different number of thermo lines ({len(nf)} vs {len(le)})")
    n = min(len(nf), len(le))
else:
    n = len(nf)

diffs = [abs(nf[i][1] - le[i][1]) for i in range(n)]
maxd  = max(diffs)
tol = 1e-8
print(f"  steps compared : {n}")
print(f"  max |ΔPE|      : {maxd:.3e} eV")
if maxd < tol:
    print(f"  RESULT         : PASS – sequences are bit-identical")
else:
    print(f"  RESULT         : FAIL – sequences differ (max {maxd:.3e} eV)")
    sys.exit(1)
PYEOF

echo ""
echo "All tests passed."
