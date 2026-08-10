# MC Swap Performance Improvements — Technical Report

**LAMMPS fork:** `thermoatoms/lammps`  
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

### Physical basis

The standard `force->pair->compute(eflag, vflag)` call computes both energy and forces.
During an MC trial, forces are irrelevant — only the scalar potential energy matters.
For ACE/PACE potentials, the force loop (back-propagation through the ACE basis
gradients to produce `neighbours_forces`) is computationally expensive. 

### LAMMPS implementation (`force.h`, `pair.h`, `pair.cpp`)

The request travels as an extra bit in the existing `eflag` word, alongside
`ENERGY_GLOBAL` and `ENERGY_ATOM`, rather than as a side-channel member poked by the
fix:

```cpp
// force.h
enum { ENERGY_NONE = 0x00, ENERGY_GLOBAL = 0x01, ENERGY_ATOM = 0x02, ENERGY_ONLY = 0x04 };
```

```cpp
// pair.cpp, Pair::ev_setup()  (reached by every pair style through ev_init)
eflag_only = eflag_global ? (eflag & ENERGY_ONLY) : 0;
```

A pair style that can skip its force loop inspects `eflag_only`; one that cannot ignores
the bit and computes forces as usual, so the flag is never wrong, only unused.  Because
`ev_setup` derives it afresh on every `compute()` call there is no state to restore
afterwards, and `hybrid` / `hybrid/scaled` forward `eflag` to their sub-styles unchanged,
so the bit reaches those without the fix having to know about them.

```cpp
// fix_atom_swap.cpp, energy_full()
int eflag = ENERGY_GLOBAL;
if (noforce_flag) eflag |= ENERGY_ONLY;
if (force->pair) force->pair->compute(eflag, vflag);
```

Forces are left undefined after an energy-only call, which is harmless here: the next MD
step recomputes them.

This is deliberately the same spelling LAMMPS itself adopted after this fork was taken
(`ENERGY_ONLY`, `Pair::eflag_only`), so code written against either tree compiles against
both.  Styles that honor it here: `pace`, `eam`, and all `grace` variants.  The eflag bit
is the only way to request energy-only evaluation: there is no side-channel flag to set,
so a caller cannot leave one stuck on.


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
aceimpl->ace->energy_only = (eflag_only != 0);
aceimpl->ace->compute_atom(i, x, type, jnum, jlist);
```

The force accumulation loop on the LAMMPS side (reading back `neighbours_forces` and
applying to `f[i]`, `f[j]`) was also guarded with `if (!eflag_only)`.  Note the two
spellings: `eflag_only` is the LAMMPS-side request bit, `ACEEvaluator::energy_only` is
the evaluator's own switch that it sets.

### Validation

Bit-identical energy trajectory and accept/reject sequence (same random seed) with and
without `noforce yes`. HPC benchmarks at 4000 atoms with the Au-Cu PACE potential show a substantially larger speedup. The speedup is independent of system size because it targets the
per-atom cost, not the total atom count.

![Figure 1](fig_speedup_force_opt.png)

*Figure 1. (Left) The energy per atom during the MCMD run with the base code and the no force optimisation. (Right) The speedup when using the no-force approach on 40 cores with 4000 atoms. The speedup is even more significant on lesser cores.*

---

## Improvement 2: Multi-Atom Swap (`swap_count N`)

**Keyword:** `swap_count N`  
**Default:** `swap_count 1` (original behaviour preserved)

### Physical basis

The original fix swaps exactly one pair of atoms per MC trial. The new `swap_count N`
keyword selects $N$ pairs simultaneously, swaps all of them atomically, evaluates the
total energy change, and accepts or rejects the entire move as a single Metropolis step.
This is a standard extension of the MC algorithm that increases the configurational
sampling rate without changing the number of energy evaluations per cycle.

### LAMMPS implementation

Changes are confined to `src/MC/fix_atom_swap.cpp/.h`. The `attempt_swap()` function
was extended to select `nswap_count` distinct atoms from each type list using the same
`random_equal` RNG (ensuring identical behaviour across MPI ranks), and to swap and
restore all of them atomically. The Metropolis criterion is applied to the total
$\Delta E$ over all $N$ pairs.

### Validation

Higher counts increase the average $|\Delta E|$ per trial and reduce the acceptance rate. Swapping more atoms in this system leads to lower acceptance.

![Figure 2](fig_swap_more_atoms.png)

*Figure 2. (Left) Energy per atom of the system with simulation time for varying number of concurrent swaps. (Right) Acceptance rate and sampling efficiency (accepted swaps per unit wall time) as a function of `swap_count` (1–4).*

---

## Improvement 3: Local Coordination-Shell Energy (`localE yes`)

**Keyword:** `localE yes`  
**Default:** `localE no`  
**Requires:** `pair_style pace`, `swap_count 1`, `semi-grand no`

### Physical basis

In the ACE formalism, the total potential energy is written as a sum of per-atom
contributions:

$$E_{\text{tot}} = \sum_{i=1}^{N} \varepsilon_i$$

where $\varepsilon_i = F_{\mu_i}(\{\rho_p^{(i)}\}) + E_0(\mu_i)$ depends only on the chemical species of atom $i$ and the positions and species of its neighbours within the cutoff radius $r_c$. Crucially, there is no explicit pairwise energy splitting — the full many-body energy of atom $i$'s environment is assigned entirely to atom $i$. This is the key property that makes local energy approximation exact within ACE.

When atom $i$ is swapped from species $\mu_i$ to $\mu_j$:

$$\Delta E = \underbrace{\Delta \varepsilon_i}_{\text{species change at }i}
           + \sum_{k \in \mathcal{N}(i)} \underbrace{\Delta \varepsilon_k}_{i\text{ appears as neighbour of }k}$$

where $\mathcal{N}(i)$ is the set of all atoms that have $i$ within their own cutoff (for ACE's full neighbour list this is the same as $i$'s neighbour list under symmetric cutoffs). The rest of the system remains unaffected. For a typical metal with $r_c \approx 5$ Å, $|\mathcal{N}(i)| \approx 50$–80. This means $\Delta E$ can be computed by rerunning `compute_atom` on ~100 atoms rather than all $N$.

### PACE implementation

Three public helper methods were added to `pair_pace.h/.cpp`:

**`compute_atom_energy(i)`** — calls `compute_atom` on a single atom with `energy_only = true`, returns the scaled `e_atom`. Reuses the existing neighbour list without any rebuild.

**`build_atom_energy_cache(eatom[], nmax)`** — iterates over all local atoms, calls `compute_atom_energy(k)` for each, fills the cache array, and returns the local energy sum. Called once per `pre_exchange()` after reneighboring.

**`compute_shell_delta(tag_i, tag_j, eatom_cached[], changed)`** — the core of the local energy path. After a type swap (types already updated, ghosts synced via `comm->forward_comm`), scans all local atoms: an atom $k$ is included in the affected set if `tag[k] == tag_i`, `tag[k] == tag_j`, or atom `tag_i`/`tag_j` appears in $k$'s neighbour list. For each affected $k$: recomputes `e_atom`, accumulates `local_dE += new_e - eatom_cached[k]`, records `(k, new_e)` for cache update on accept. Returns the local $\Delta E$ contribution; caller does one `MPI_Allreduce`.

### LAMMPS implementation

**Cache management:**
- `eatom_cached[]` is allocated at `init()` time and grown as needed.
- `build_eatom_cache()` in the fix calls `pace->build_atom_energy_cache()` then `MPI_Allreduce` for the global sum, replacing `energy_full()` at the top of `pre_exchange()`.

**Per-trial path in `attempt_swap()`:**
1. Swap types on owning ranks
2. `comm->forward_comm(this)` — ghost types updated on all ranks
3. Tag broadcast: `MPI_Allreduce` of `{tag[i], tag[j]}` so all ranks know which two global atoms were swapped
4. `pace->compute_shell_delta(...)` — returns local $\Delta E$
5. `MPI_Allreduce` → global $\Delta E$, Metropolis test
6. Accept: update `eatom_cached` for affected atoms, `energy_stored += delta`
7. Reject: restore types, one `forward_comm` to re-sync ghosts

The total MPI traffic per trial is two tag integers broadcast plus one double reduce replacing the expensive global pair compute and its PE reduction with a trivial local scan of ~100 atoms.

### Validation

HPC benchmarks at 4000 atoms with the Au-Cu PACE potential confirm that the acceptance rate is unchanged relative to the full-system reference, demonstrating that the local approximation is **exact** for ACE (no bias introduced):

![Figure 3](fig_localE_acceptance.png)

*Figure 3. (Left) Energy per atom with simulation time for base and optimised methods (Right) MC acceptance rate for `localE yes` compared to the full-system reference (`noforce yes`). Benchmarked on 4000-atom Au-Cu at 800 K using the PACE potential on the HPC cluster. The distributions are bit-identical, confirming that the local coordination-shell approximation introduces no bias for ACE potentials.*

The wall-time speedup at 4000 atoms scales as predicted — approximately linear in $N$
relative to `noforce yes`, confirming the $O(Z^2)$ vs $O(N \cdot Z)$ cost reduction:

![Figure 4](fig_localE_speedup.png)

*Figure 4. (Left) Wall-time speedup of `localE yes` relative to both `noforce yes` and the unmodified baseline. (Right) Scaling with number of cores for base and optimised methods. There is a slight overhead due to communication for the optimised method.*

---

## Improvement 4: Improved `pair_style hybrid/scaled` (PACE + PACE)

### Physical basis

Alchemical free energy calculations in calphy use `pair_style hybrid/scaled` with two PACE sub-styles scaled by complementary λ-dependent variables:

```lammps
pair_style  hybrid/scaled v_flambda pace v_blambda pace
pair_coeff  * * pace 1 AuCu.yace Au Au
pair_coeff  * * pace 2 AuCu.yace Au Cu
```

This setup continuously morphs the interatomic potential from one chemical state to another as λ varies from 0 to 1. MC atom swaps must remain efficient across the full λ path.

The total scaled energy is:

$$E_{\text{tot}} = \lambda \sum_i \varepsilon_i^{(A)} + (1-\lambda) \sum_i \varepsilon_i^{(B)}$$

Three new public helpers were added to `PairPACE`:

- **`get_affected_local_atoms(tag_i, tag_j, affected)`** — extracts the coordination-shell scan into a reusable primitive shared across all code paths
- **`accumulate_atom_energies(scale, eatom[], nmax)`** — adds `scale × e_atom` into an existing cache array, enabling multi-style accumulation without a temporary buffer
- **`is_type_invariant(t1, t2)`** — returns true when both LAMMPS types map to the same ACE species index; used to detect the type-invariant sub-style at `init()` time

A new member `pace_substyles` (`std::vector<std::pair<PairPACE*, double>>`) is populated at `init()` for both the plain-PACE and hybrid paths, so all downstream code (`build_eatom_cache`, `attempt_swap`, `compute_shell_delta`) uses a single unified loop with no special-casing.

When `is_type_invariant` detects that one sub-style maps both LAMMPS atom types to the same ACE species (the typical alchemical endpoint sub-style, e.g. `Au Au`), the fix maintains two independent per-style energy caches `eatom_sA[]` and `eatom_sB[]` instead of a single blended cache. The blended `eatom_cached[i]` is then:

```
eatom_cached[i] = scale_A * eatom_sA[i] + scale_B * eatom_sB[i]
```

The key benefit: swapping two atoms cannot change `eatom_sA[i]` for the type-invariant sub-style (the ACE species assignment is identical for both LAMMPS types). The `attempt_swap` trial therefore calls `compute_atom_energy` only for the type-sensitive sub-style on the ~2Z affected atoms, and skips the invariant sub-style entirely. On acceptance, only `eatom_sB` is updated.


**`noforce yes` propagation through hybrid**

`hybrid` and `hybrid/scaled` pass `eflag` through to their sub-styles unchanged, so the
`ENERGY_ONLY` bit set once in `energy_full()` reaches every sub-style on its own. The
force-loop skip therefore operates on both PACE sub-styles simultaneously, giving the
same ~2–3× MC timer reduction measured for the single-PACE case.

### Validation

Three validation cases were run on the 4000-atom Au-Cu test system. The evaluated energy differences between the two sub-styles in the hybrid pair style is same with and without the optimisation.


![Figure 5](fig_speedup_hybrid_opt.png)

_Figure 5. (Left) energy difference between the two pair styles with simulation time for base and optimised methods (Right) Wall-time speedup of the optimised method in comparison to the base approach.

---

## Summary of Files Changed

### `thermoatoms/lammps` — branch `energy_eval`

| File | Change |
|---|---|
| `src/force.h` | `ENERGY_ONLY = 0x04` eflag bit |
| `src/pair.h`, `src/pair.cpp` | `eflag_only` derived in `ev_setup` / cleared in `ev_unset` |
| `src/MANYBODY/pair_eam.cpp` | Guard force loop with `if (!eflag_only)` |
| `src/ML-PACE/pair_grace*.cpp` | Energy-only path triggered by `eflag_only` as well as by `extract("compute_energy_only")` |
| `src/ML-PACE/pair_pace.h` | 5 public local-energy helpers; `get_affected_local_atoms`, `accumulate_atom_energies` added for hybrid support |
| `src/ML-PACE/pair_pace.cpp` | Implement all helpers; refactor `compute_shell_delta` to call `get_affected_local_atoms` |
| `src/pair_hybrid_scaled.h` | `friend class FixAtomSwap` to allow reading `scaleval[]` |
| `src/MC/fix_atom_swap.h` | `swap_count`, `noforce`, `localE` members; `pace_substyles` vector; `build_eatom_cache()` |
| `src/MC/fix_atom_swap.cpp` | All three keywords; hybrid/scaled support; `noforce` sets `ENERGY_ONLY` in `energy_full()`; PACE-only localE paths behind `LMP_ATOM_SWAP_PACE` so `PKG_MC` still builds without ML-PACE |
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

