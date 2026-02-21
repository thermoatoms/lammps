#!/bin/bash
# Test the split-cache path:
#   sub-style 1: "Au Au" (type-invariant) → split_cache_flag=1
#   sub-style 2: "Au Cu" (type-aware)
#
# Correctness checks:
#   1) lambda=1 (hybrid/scaled 1 0): PE should equal plain-pace "Au Au"  (REF_AA)
#   2) lambda=0 (hybrid/scaled 0 1): PE should equal plain-pace "Au Cu"  (REF_AC)
#   3) lambda=0.5: split-cache activated, sanity check (no crash, finite PE)

set -euo pipefail
cd "$(dirname "$0")"

LMP=${LMP:-../build/lmp}
TOL=1e-8

run() {
    local label=$1; shift
    echo ">>> Running $label ..."
    "$LMP" "$@" -screen none -log "log_${label}.lammps"
    echo "    done."
}

run REF_AA  -in lmp_auau_ref.in
run SC_LAM1 -var lambda 1.0 -in lmp_splitcache.in
sed 's/SWAP_KEYWORD/localE yes/' lmp_pace.in > lmp_pace_auce_ref.in
run REF_AC  -in lmp_pace_auce_ref.in
run SC_LAM0 -var lambda 0.0 -in lmp_splitcache.in
run SC_MID  -var lambda 0.5 -in lmp_splitcache.in

echo ""

python3 - <<'PYEOF'
import sys, os

def parse_pe(logfile):
    pes = []
    in_run = False
    pe_col = None
    with open(logfile) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Step"):
                in_run = True
                cols = stripped.split()
                pe_col = cols.index("PotEng")
                continue
            if in_run and pe_col is not None:
                parts = stripped.split()
                if parts and parts[0].lstrip("-").lstrip("+").isdigit():
                    try:
                        pes.append(float(parts[pe_col]))
                    except (ValueError, IndexError):
                        pass
    return pes

ref_aa  = parse_pe("log_REF_AA.lammps")
sc_lam1 = parse_pe("log_SC_LAM1.lammps")
ref_ac  = parse_pe("log_REF_AC.lammps")
sc_lam0 = parse_pe("log_SC_LAM0.lammps")
sc_mid  = parse_pe("log_SC_MID.lammps")

def compare(name, a, b, tol=1e-8):
    n = min(len(a), len(b))
    if n == 0:
        print(f"  {name}: no data!")
        return False
    diffs = [abs(a[i] - b[i]) for i in range(n)]
    maxd = max(diffs)
    ok = maxd < tol
    print(f"=== {name} ===")
    print(f"  steps compared : {n}")
    print(f"  max |DeltaPE|  : {maxd:.3e} eV")
    print(f"  RESULT         : {'PASS' if ok else 'FAIL'} (tol = {tol:.0e} eV)")
    print()
    return ok

ok1 = compare("SC_LAM1 vs REF_AA  (hybrid/scaled lam=1 vs plain pace Au Au)", sc_lam1, ref_aa)
ok2 = compare("SC_LAM0 vs REF_AC  (hybrid/scaled lam=0 vs plain pace Au Cu)", sc_lam0, ref_ac)

# sanity: SC_MID produced finite PE values
ok3 = len(sc_mid) > 0 and all(abs(e) < 1e10 for e in sc_mid)
print("=== SC_MID sanity (lambda=0.5, finite PE, no crash) ===")
print(f"  steps recorded : {len(sc_mid)}")
print(f"  RESULT         : {'PASS' if ok3 else 'FAIL'}")
print()

if ok1 and ok2 and ok3:
    print("All split-cache tests passed.")
    sys.exit(0)
else:
    print("SOME TESTS FAILED.")
    sys.exit(1)
PYEOF
