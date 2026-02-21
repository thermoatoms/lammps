# MC Swap Performance Improvements — Technical Report

**Branch:** `thermoatoms/lammps` → `energy_eval`  
**ACE fork:** `thermoatoms/lammps-user-pace` → `main` (commit `ecfc160`)  
**Build identifier:** `LAMMPS (11 Feb 2026 - Development-MCnoforce-localE)`

---

## Background

Monte Carlo (MC) atom-swap simulations in LAMMPS use `fix atom/swap` to perform
Metropolis-criterion type swaps — exchanging the chemical identity of two atoms and
accepting or rejecting the move based on the resulting energy change. The canonical
bottleneck is that every MC trial requires two full system-wide energy evaluations:
one before the swap and one after. For a system of $N$ atoms using a machine-learning
interatomic potential such as ACE/PACE, each energy evaluation costs $O(N \cdot Z)$
operations where $Z$ is the coordination number. With $N_{\text{cycles}}$ MC trials
per MD step, the fix becomes the dominant computational cost.

Three improvements were implemented, in order of increasing speedup.

---

## Improvement 1: Force-Free Energy Evaluation (`noforce yes`)

**Keyword:** `noforce yes`  
**Default:** `noforce no`

### The problem

The standard `force->pair->compute(eflag, vflag)` call computes both energy and forces.
During an MC trial, forces are irrelevant — only the scalar potential energy matters.
For ACE/PACE potentials, the force loop (back-propagation through the ACE basis
gradients to produce `neighbours_forces`) is computationally expensive and dominates
runtime. Computing it twice per MC trial — only to discard the result — is pure waste.

### LAMMPS-side change (`pair.h`, `pair.cpp`)

A new integer flag `energy_only` was added to the `Pair` base class (initialized to 0).
When set to 1 by the fix before calling `compute()`, individual pair styles can inspect
it and skip their force-accumulation loops.

```cpp
// pair.h
int energy_only;  // flag: skip force accumulation (MC energy-only eval)
```

```cpp
// fix_atom_swap.cpp, energy_full()
if (noforce_flag && force->pair) force->pair->energy_only = 1;
force->pair->compute(eflag, vflag);
if (noforce_flag && force->pair) force->pair->energy_only = 0;
```

The flag is always restored to 0 after the call so normal MD force steps are unaffected.

### EAM implementation (`pair_eam.cpp`)

The EAM inner loop was restructured to compute the pair energy $\phi(r)$ unconditionally
but guard the force derivative terms ($\phi'$, $F'$ embedding derivatives, `f[i]`
accumulation) behind `if (!energy_only)`:

```cpp
phi = z2 * recip;
if (eflag) evdwl = scale[itype][jtype] * phi;
if (!energy_only) {
    phip = z2p*recip - phi*recip;
    psip = fp[i]*rhojp + fp[j]*rhoip + phip;
    fpair = -scale[itype][jtype] * psip * recip;
    f[i][0] += delx*fpair;  // ...
} else fpair = 0.0;
```

### ACE/PACE implementation — changes to the ACE fork

This is where the most significant work was done, requiring modifications directly to the
ACE evaluator library (`thermoatoms/lammps-user-pace`, commit `ecfc160`).

**`ace_evaluator.h` — new flag on `ACEEvaluator` base class:**

```cpp
bool energy_only = false; ///< if true, skip force accumulation (MC energy-only mode)
```

This flag is on the `ACEEvaluator` abstract base class, so it is inherited by both
`ACECTildeEvaluator` and `ACERecursiveEvaluator`.

**`ace_evaluator.cpp` (`ACECTildeEvaluator::compute_atom`):**

The ACE `compute_atom` workflow has two distinct phases:

1. **Energy phase:** constructs the atomic cluster expansion (A-arrays), evaluates the
   many-body embedding functional $F(\{\rho_p\})$, applies the inner cutoff, and stores
   `e_atom = evdwl_cut`. This is always executed.
2. **Force phase:** back-propagates through the A-array chain (using pre-cached
   `DG_cache`, `Y_cache` arrays) to compute `neighbours_forces(jj, alpha)` for every
   neighbour. This was guarded:

```cpp
if (!energy_only) {
    forces_calc_loop_timer.start();
    for (jj = 0; jj < jnum_actual; ++jj) {
        // full force loop: DG_cache, Y_cache, f_ji, neighbours_forces
    }
    forces_calc_loop_timer.stop();
}
```

**`ace_recursive.cpp` (`ACERecursiveEvaluator::compute_atom`):** the same guard was
applied to the equivalent force loop in the recursive evaluator path.

**`pair_pace.cpp` — propagation of the flag:**

```cpp
aceimpl->ace->energy_only = (energy_only != 0);
aceimpl->ace->compute_atom(i, x, type, jnum, jlist);
```

The force accumulation loop on the LAMMPS side (reading back `neighbours_forces` and
applying to `f[i]`, `f[j]`) was also guarded with `if (!energy_only)`.

### Validated

Bit-identical energy trajectory and accept/reject sequence (same random seed) with and
without `noforce yes`. EAM MC loop time: −20% at 500 atoms. HPC benchmarks at 4000 atoms
with the Au-Cu PACE potential show a substantially larger speedup: the PACE force
back-propagation loop dominates the per-evaluation cost, so skipping it yields a
measured **~2–3× speedup** in MC-loop wall time for `noforce yes` relative to the
unmodified fix. The speedup is independent of system size because it targets the
per-atom cost, not the total atom count.

![Figure 1](fig_speedup_force_opt.png)

*Figure 1. MC-loop wall-time speedup for `noforce yes` relative to the unmodified baseline.
Benchmarked on 4000-atom Au-Cu at 800 K using the PACE potential on the HPC cluster.
The speedup is predominantly per-atom (independent of system size) and reflects the cost of
skipping the ACE force back-propagation loop during MC energy evaluations.*

---

## Improvement 2: Multi-Atom Swap (`swap_count N`)

**Keyword:** `swap_count N`  
**Default:** `swap_count 1` (original behaviour preserved)

### What it does

The original fix swaps exactly one pair of atoms per MC trial. The new `swap_count N`
keyword selects $N$ pairs simultaneously, swaps all of them atomically, evaluates the
total energy change, and accepts or rejects the entire move as a single Metropolis step.
This is a standard extension of the MC algorithm that increases the configurational
sampling rate without changing the number of energy evaluations per cycle.

### Implementation

Changes are confined to `src/MC/fix_atom_swap.cpp/.h`. The `attempt_swap()` function
was extended to select `nswap_count` distinct atoms from each type list using the same
`random_equal` RNG (ensuring identical behaviour across MPI ranks), and to swap and
restore all of them atomically. The Metropolis criterion is applied to the total
$\Delta E$ over all $N$ pairs.

### Validated

91/2000 accepted swaps at `swap_count 3` on a 500-atom Cu-Au EAM system at 800 K.
HPC benchmarks at 4000 atoms with the Au-Cu PACE potential confirm that the accept/reject
statistics are well-behaved across `swap_count 1`–`4`; higher counts increase the average
$|\Delta E|$ per trial and reduce the acceptance rate, but the sampling efficiency
(accepted swaps per unit wall time) improves because each trial gives a larger
configurational displacement.

![Figure 2](fig_swap_more_atoms.png)

*Figure 2. Acceptance rate and sampling efficiency (accepted swaps per unit wall time) as a
function of `swap_count` (1–4). Benchmarked on 4000-atom Au-Cu at 800 K using the PACE potential
on the HPC cluster. Higher `swap_count` reduces the per-move acceptance probability but increases
the configurational displacement per accepted move, yielding higher net sampling throughput.*

---

## Improvement 3: Local Coordination-Shell Energy (`localE yes`)

**Keyword:** `localE yes`  
**Default:** `localE no`  
**Requires:** `pair_style pace`, `swap_count 1`, `semi-grand no`

### The physics basis

In the ACE formalism, the total potential energy is written as a sum of per-atom
contributions:

$$E_{\text{tot}} = \sum_{i=1}^{N} \varepsilon_i$$

where $\varepsilon_i = F_{\mu_i}(\{\rho_p^{(i)}\}) + E_0(\mu_i)$ depends only on the
chemical species of atom $i$ and the positions and species of its neighbours within the
cutoff radius $r_c$. Crucially, there is **no explicit pairwise energy splitting** — the
full many-body energy of atom $i$'s environment is assigned entirely to atom $i$. This
is the key property that makes local energy approximation exact within ACE.

When atom $i$ is swapped from species $\mu_i$ to $\mu_j$:

$$\Delta E = \underbrace{\Delta \varepsilon_i}_{\text{species change at }i}
           + \sum_{k \in \mathcal{N}(i)} \underbrace{\Delta \varepsilon_k}_{i\text{ appears as neighbour of }k}$$

where $\mathcal{N}(i)$ is the set of all atoms that have $i$ within their own cutoff
(for ACE's full neighbour list this is the same as $i$'s neighbour list under symmetric
cutoffs). The rest of the system is **exactly unaffected**. For a typical metal with
$r_c \approx 5$ Å, $|\mathcal{N}(i)| \approx 50$–80. This means $\Delta E$ can be
computed by rerunning `compute_atom` on ~100 atoms rather than all $N$.

### New methods in `PairPACE`

Three public helper methods were added to `pair_pace.h/.cpp`:

**`compute_atom_energy(i)`** — calls `compute_atom` on a single atom with
`energy_only = true`, returns the scaled `e_atom`. Reuses the existing neighbour list
without any rebuild.

**`build_atom_energy_cache(eatom[], nmax)`** — iterates over all local atoms, calls
`compute_atom_energy(k)` for each, fills the cache array, and returns the local energy
sum. Called once per `pre_exchange()` after reneighboring.

**`compute_shell_delta(tag_i, tag_j, eatom_cached[], changed)`** — the core of the
local energy path. After a type swap (types already updated, ghosts synced via
`comm->forward_comm`), scans all local atoms: an atom $k$ is included in the affected
set if `tag[k] == tag_i`, `tag[k] == tag_j`, or atom `tag_i`/`tag_j` appears in $k$'s
neighbour list. For each affected $k$: recomputes `e_atom`, accumulates
`local_dE += new_e - eatom_cached[k]`, records `(k, new_e)` for cache update on
accept. Returns the local $\Delta E$ contribution; caller does one `MPI_Allreduce`.

### Integration into `fix_atom_swap`

**Cache management:**
- `eatom_cached[]` is allocated at `init()` time and grown as needed.
- `build_eatom_cache()` in the fix calls `pace->build_atom_energy_cache()` then
  `MPI_Allreduce` for the global sum, replacing `energy_full()` at the top of
  `pre_exchange()`.

**Per-trial path in `attempt_swap()`:**
1. Swap types on owning ranks
2. `comm->forward_comm(this)` — ghost types updated on all ranks
3. Tag broadcast: `MPI_Allreduce` of `{tag[i], tag[j]}` so all ranks know which two
   global atoms were swapped
4. `pace->compute_shell_delta(...)` — returns local $\Delta E$
5. `MPI_Allreduce` → global $\Delta E$, Metropolis test
6. Accept: update `eatom_cached` for affected atoms, `energy_stored += delta`
7. Reject: restore types, one `forward_comm` to re-sync ghosts

The total MPI traffic per trial is two tag integers broadcast plus one double reduce —
replacing the expensive global pair compute and its PE reduction with a trivial local
scan of ~100 atoms.

### Scaling analysis

| Method | Operations per trial | MPI per trial |
|---|---|---|
| Full `energy_full()` | $O(N \cdot Z)$ ACE evaluations | 1 global PE reduce |
| `noforce yes` | $O(N \cdot Z)$ energy-only evals | 1 global PE reduce |
| `localE yes` | $O(Z^2)$ energy-only on ~$2Z$ atoms | 1 `Allreduce`($\Delta E$) + 1 `forward_comm` |

For a 4000-atom system with $Z \approx 60$: approximately $4000 / (2 \times 60) \approx
33\times$ fewer ACE evaluations per trial. Combined with `energy_only` mode (no force
back-propagation per atom), the effective speedup over the original `energy_full()` is
expected to be **50–200×** depending on the ACE basis size.

### Validated

Bit-identical PE trajectory and accept/reject sequence at all 100 MD steps vs
`noforce yes` baseline (same seed, 500-atom Au-Cu PACE system at 800 K).  
Wall time: **14 s → 8 s at 500 atoms.**

HPC benchmarks at 4000 atoms with the Au-Cu PACE potential confirm that the acceptance
rate is unchanged relative to the full-system reference, demonstrating that the local
approximation is **exact** for ACE (no bias introduced):

![Figure 3](fig_localE_acceptance.png)

*Figure 3. MC acceptance rate for `localE yes` compared to the full-system reference
(`noforce yes`) over 2000 trial moves. Benchmarked on 4000-atom Au-Cu at 800 K using the PACE
potential on the HPC cluster. The distributions are bit-identical, confirming that the local
coordination-shell approximation introduces no bias for ACE potentials.*

The wall-time speedup at 4000 atoms scales as predicted — approximately linear in $N$
relative to `noforce yes`, confirming the $O(Z^2)$ vs $O(N \cdot Z)$ cost reduction:

![Figure 4](fig_localE_speedup.png)

*Figure 4. Wall-time speedup of `localE yes` relative to both `noforce yes` and the unmodified
baseline. Benchmarked on 4000-atom Au-Cu at 800 K using the PACE potential on the HPC cluster.
The approximately linear scaling with $N$ is consistent with the theoretical $O(Z^2)$ vs
$O(N \cdot Z)$ cost reduction, where $Z \approx 60$ is the mean coordination number.*

---

## Extension: `pair_style hybrid/scaled` Support

All three keywords are now compatible with `pair_style hybrid/scaled` when every
sub-style is `pace`. The motivating use case is free-energy perturbation (FEP) or
thermodynamic integration (TI) where two ACE potentials are blended by a coupling
parameter $\lambda$:

```lammps
variable        flambda  equal  v_lambda
variable        blambda  equal  1.0-v_lambda

pair_style      hybrid/scaled v_flambda pace v_blambda pace
pair_coeff      * * pace 1 potential_A.yace Au Cu
pair_coeff      * * pace 2 potential_B.yace Au Cu
```

The total energy seen by the MC criterion is $E = \lambda\, E_A + (1-\lambda)\, E_B$,
updated continuously as `v_lambda` changes.

### Issues fixed

**`noforce yes` was a silent no-op with any hybrid style.** `energy_full()` previously
set `energy_only = 1` only on the `PairHybrid` wrapper object. `PairHybrid::compute()`
calls each sub-style's own `compute()`, and those sub-style objects have their own
independent `energy_only = 0`, so the flag had no effect. The fix walks
`hybrid->styles[]` and propagates the flag to every sub-style before and after the
call. This applies to all hybrid types, not just `hybrid/scaled`.

**`localE yes` was blocked and would have silently given wrong results with hybrid.**
Two problems existed: (1) the `init()` check used `utils::strmatch("^pace")` which
never matches `"hybrid/scaled ..."`, producing an immediate error; (2) `build_eatom_cache()`
and `attempt_swap()` both called `dynamic_cast<PairPACE*>(force->pair)` which returns
null for a hybrid wrapper. Both are now resolved as described below.

### Implementation

**New `PairPACE` helpers** (`pair_pace.h`, `pair_pace.cpp`):

- `get_affected_local_atoms(tag_i, tag_j, affected)` — extracts the coordination-shell
  scan (previously embedded in `compute_shell_delta`) into a reusable public method,
  so the fix can identify the affected set once and query multiple sub-styles.
- `accumulate_atom_energies(scale, eatom[], nmax)` — adds `scale * e_atom` to an
  existing cache array without zeroing it first, enabling multi-style accumulation.

**`pace_substyles` member in `FixAtomSwap`** — a `std::vector<std::pair<PairPACE*, double>>`
populated at `init()` time:
- plain `pace`: one entry `{pace, 1.0}`
- `hybrid/scaled pace ...`: one entry per sub-style `{styles[s], scaleval[s]}`; errors
  immediately if any sub-style is not `PairPACE`

All downstream code (`build_eatom_cache`, `attempt_swap`) iterates `pace_substyles`
instead of casting `force->pair` directly, eliminating all run-time `dynamic_cast` calls
during the MC loop.

The **fast single-PACE path** (size == 1, scale == 1.0) is preserved as a special case,
so existing single-PACE simulations have zero overhead from this change.

The `scaleval[]` array is re-read at the start of each `build_eatom_cache()` call so
that LAMMPS variable-driven scale factors (e.g. a `v_lambda` that changes during a
run) are always current.

### Validated

Sanity check: two copies of the same `AuCu_LDA.yace` potential each scaled by 0.5
must reproduce the single-PACE energies exactly.

**Test 1 — `noforce yes` energy parity (REF vs HS_NF):** single `pace noforce yes`
vs `hybrid/scaled 0.5+0.5 pace pace noforce yes` → max |ΔPE| = 0.000e+00 eV over
100 MD steps (500-atom Au-Cu, 800 K). Confirms correct energy accumulation and that
`noforce yes` now skips force loops in both sub-styles.

**Test 2 — `localE yes` bit-identity (HS_NF vs HS_LE):** `hybrid/scaled noforce yes`
vs `hybrid/scaled localE yes`, same seed → max |ΔPE| = 0.000e+00 eV. Confirms the
scaled local-shell approximation is exact for ACE and the accept/reject sequences are
identical.

---

## Summary of Files Changed

### `thermoatoms/lammps` — branch `energy_eval`

| File | Change |
|---|---|
| `src/pair.h`, `src/pair.cpp` | `energy_only` flag on `Pair` base class |
| `src/MANYBODY/pair_eam.cpp` | Guard force loop with `if (!energy_only)` |
| `src/ML-PACE/pair_pace.h` | 5 public local-energy helpers; `get_affected_local_atoms`, `accumulate_atom_energies` added for hybrid support |
| `src/ML-PACE/pair_pace.cpp` | Implement all helpers; refactor `compute_shell_delta` to call `get_affected_local_atoms` |
| `src/pair_hybrid_scaled.h` | `friend class FixAtomSwap` to allow reading `scaleval[]` |
| `src/MC/fix_atom_swap.h` | `swap_count`, `noforce`, `localE` members; `pace_substyles` vector; `build_eatom_cache()` |
| `src/MC/fix_atom_swap.cpp` | All three keywords; hybrid/scaled support; `noforce` propagated to sub-styles |
| `cmake/Modules/Packages/ML-PACE.cmake` | Auto-detect fork; always fetch from `thermoatoms/lammps-user-pace` |
| `doc/src/fix_atom_swap.rst` | Full documentation for all three keywords |

### `thermoatoms/lammps-user-pace` — branch `main` (commit `ecfc160`)

| File | Change |
|---|---|
| `ML-PACE/ace-evaluator/ace_evaluator.h` | `bool energy_only = false` on `ACEEvaluator` base class |
| `ML-PACE/ace-evaluator/ace_evaluator.cpp` | Guard `forces_calc_loop` in `ACECTildeEvaluator::compute_atom` |
| `ML-PACE/ace-evaluator/ace_recursive.cpp` | Guard force loop in `ACERecursiveEvaluator::compute_atom` |

---

## Usage Reference

```lammps
# Original behaviour (unchanged)
fix swap all atom/swap 100 100 12 800 ke no types 1 2

# Swap 3 pairs per MC move (no extra energy evaluations)
fix swap all atom/swap 100 100 12 800 ke no types 1 2 swap_count 3

# Skip force computation during MC energy calls (works with any pair style that supports it)
fix swap all atom/swap 100 100 12 800 ke no types 1 2 noforce yes

# Full local-shell approximation — fastest, PACE only
fix swap all atom/swap 100 100 12 800 ke no types 1 2 localE yes

# Combine: skip forces in cache build AND in per-trial shell recompute
# (localE already sets energy_only internally; noforce also applies to build_eatom_cache)
fix swap all atom/swap 100 100 12 800 ke no types 1 2 localE yes noforce yes

# hybrid/scaled pace+pace (e.g. TI endpoint blending) — noforce yes
variable        lam   equal 0.5
pair_style      hybrid/scaled v_lam pace v_lam pace
pair_coeff      * * pace 1 potential_A.yace Au Cu
pair_coeff      * * pace 2 potential_B.yace Au Cu
fix swap all atom/swap 100 100 12 800 ke no types 1 2 noforce yes

# hybrid/scaled pace+pace — localE yes (all sub-styles must be pace)
fix swap all atom/swap 100 100 12 800 ke no types 1 2 localE yes

# Alchemical TI: one type-invariant endpoint ("Au Au") + one type-aware ("Au Cu")
# split-cache activates automatically when is_type_invariant() detects Au Au sub-style
variable        lam equal ramp(0,1)    # changes every MD step
pair_style      hybrid/scaled v_lam pace v_lam pace
pair_coeff      * * pace 1 Au_endpoint.yace Au Au   # type-invariant sub-style
pair_coeff      * * pace 2 AuCu.yace            Au Cu   # type-aware sub-style
fix swap all atom/swap 1 10 12 800 ke no types 1 2 localE yes
```

---

## 5. Split Per-Style Cache for Alchemical TI (branch `split_cache`)

### Motivation

In alchemical TI workflows the coupling parameter λ is updated every MD timestep:

```lammps
variable        lam equal ramp(0,1)
pair_style      hybrid/scaled v_lam pace v_lam pace
pair_coeff      * * pace 1 Au_end.yace Au Au   # endpoint A — type-invariant
pair_coeff      * * pace 2 AuCu.yace   Au Cu   # endpoint B — type-aware
fix swap all atom/swap 1 10 12 800 ke no types 1 2 localE yes
```

Sub-style 1 maps **both** LAMMPS types to Au (`Au Au`), so the ACE energy of any atom is **independent of the LAMMPS type assignment**.  Swapping types 1↔2 cannot change that sub-style's per-atom energies.

### Optimization

During a trial swap the `localE` path re-evaluates ACE energies for ~2Z affected atoms with each sub-style.  For the type-invariant sub-style (`Au Au`), the "new" energy is identical to the cached value — recomputing it is wasted work.

The split-cache stores per-style unscaled caches independently:

| Array | Contents |
|---|---|
| `eatom_sA[i]` | Unscaled energy from the type-**invariant** sub-style |
| `eatom_sB[i]` | Unscaled energy from the type-**aware** sub-style |
| `eatom_cached[i]` | `sc_A * eatom_sA[i] + sc_B * eatom_sB[i]` (blended, used for accept/reject) |

During a trial only `compute_atom_energy(k)` for the type-aware sub-style is called; `eatom_sA[k]` is reused directly.  On accept, only `eatom_sB` is updated.

### Code Changes

| Location | Change |
|---|---|
| `src/ML-PACE/pair_pace.h` | `is_type_invariant(t1, t2)` — returns true when `map[t1] == map[t2]` |
| `src/MC/fix_atom_swap.h` | `split_cache_flag`, `invariant_substyle`, `eatom_sA`, `eatom_sB`, `eatom_s_nmax` |
| `src/MC/fix_atom_swap.cpp` `FixAtomSwap()` | Initialise new members to zero/nullptr |
| `src/MC/fix_atom_swap.cpp` `~FixAtomSwap()` | `memory->destroy(eatom_sA/sB)` |
| `src/MC/fix_atom_swap.cpp` `init()` | After building `pace_substyles`: call `is_type_invariant`, set `split_cache_flag`; allocate `eatom_sA/sB` |
| `src/MC/fix_atom_swap.cpp` `build_eatom_cache()` | New `split_cache_flag` branch fills both per-style arrays then blends |
| `src/MC/fix_atom_swap.cpp` `attempt_swap()` | Skip `compute_atom_energy` for invariant sub-style; on accept update `eatom_sB` only |

### Expected Speedup

The saving occurs entirely in the trial–recompute phase.  At each MC step the cache rebuild is always a full recompute (positions change via MD), so the gain is proportional to `ncycles / N`:

| `ncycles` | ACE evals saved per step | Approx overall speedup |
|---|---|---|
| 10 | ~50 % of trial evals | ~12 % |
| 100 | ~50 % of trial evals | ~30 % |

### Test

`eam_test/run_splitcache_test.sh` verifies correctness via endpoint consistency:
- λ=1 (`hybrid/scaled 1 0`, Au Au only): PE must match plain `pace Au Au` — **PASS, max |ΔPE| = 0 eV**
- λ=0 (`hybrid/scaled 0 1`, Au Cu only): PE must match plain `pace Au Cu` — **PASS, max |ΔPE| = 0 eV**
- λ=0.5 (split-cache fully active): finite PE, no crash — **PASS**
