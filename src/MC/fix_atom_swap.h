/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(atom/swap,FixAtomSwap);
// clang-format on
#else

#ifndef LMP_FIX_ATOM_SWAP_H
#define LMP_FIX_ATOM_SWAP_H

#include "fix.h"

#include <unordered_map>
#include <utility>
#include <vector>

namespace LAMMPS_NS {

class FixAtomSwap : public Fix {
 public:
  FixAtomSwap(class LAMMPS *, int, char **);
  ~FixAtomSwap() override;
  int setmask() override;
  void init() override;
  void pre_exchange() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;
  int modify_param(int, char **) override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void *extract(const char *, int &) override;

  int image(int *&, double **&) override;

 private:
  int nevery, seed;
  int temp_var_flag;    // 1 if temperature is backed by an equal-style variable
  int temp_var_index;   // variable index for temperature
  char *temp_var_name;  // variable name (without "v_" prefix)
  int ke_flag;            // yes = conserve ke, no = do not conserve ke
  int semi_grand_flag;    // yes = semi-grand canonical, no = constant composition
  int ncycles;
  int nswap_count;             // number of atom pairs to swap per MC move
  int noforce_flag;            // 1 = skip force accumulation during MC energy evals
  int local_energy_flag;       // 1 = use local shell energy (PACE only, nswap_count=1)
  double *eatom_cached;        // cached per-local-atom ACE energies
  int eatom_cached_nmax;       // allocated size of eatom_cached
  // sub-styles for localE with hybrid/scaled pace: (PairPACE*, scale)
  // for plain pace this has exactly one entry with scale=1.0
  std::vector<std::pair<class PairPACE *, double>> pace_substyles;

  // split per-style cache: stores unscaled per-atom energies for each sub-style
  // active when pace_substyles.size()==2 and one sub-style is type-invariant
  int split_cache_flag;   // 1 = split cache active
  int invariant_substyle; // index (0 or 1) of the type-invariant sub-style; -1 if none
  double *eatom_sA;       // unscaled per-atom energies for sub-style indexed by invariant_substyle
  double *eatom_sB;       // unscaled per-atom energies for the type-aware sub-style
  int eatom_s_nmax;       // allocated length of eatom_sA / eatom_sB
  int niswap, njswap;                  // # of i,j swap atoms on all procs
  int niswap_local, njswap_local;      // # of swap atoms on this proc
  int niswap_before, njswap_before;    // # of swap atoms on procs < this proc
  int nswap;                           // # of swap atoms on all procs
  int nswap_local;                     // # of swap atoms on this proc
  int nswap_before;                    // # of swap atoms on procs < this proc
  class Region *region;                // swap region
  char *idregion;                      // swap region id

  int mc_active;    // 1 during MC trials, otherwise 0

  int nswaptypes, nmutypes;
  int *type_list;
  double *mu;
  int *mu_var_flag;     // 1 if mu[i] is backed by an equal-style variable
  int *mu_var_index;    // variable index (from input->variable->find)
  char **mu_var_names;  // variable names as given in input (without "v_" prefix)

  // adaptive mu stepping via susceptibility
  int adapt_flag;           // 1 if adaptive stepping is active
  int adapt_type;           // atom type index (1-based) whose mu is driven
  double adapt_dX;          // target composition step per adaptation
  int adapt_every;          // number of MC blocks between mu updates (K)
  double adapt_dmu_max;     // max |delta_mu| per step (clamp)
  int adapt_counter;        // blocks accumulated since last mu update
  double adapt_sum_N2;      // running sum of N_adapt_type
  double adapt_sum_N2sq;    // running sum of N_adapt_type^2
  double adapt_x_current;   // current mean composition (for output)
  double adapt_chi_current; // current susceptibility dX/dmu (for output)
  double adapt_mu_current;  // current adaptive mu value (for output)

  double nswap_attempts;
  double nswap_successes;

  bool unequal_cutoffs;

  int atom_swap_nmax;
  double beta;
  double *qtype, *mtype;
  double energy_stored;
  double **sqrt_mass_ratio;
  int *local_swap_iatom_list;
  int *local_swap_jatom_list;
  int *local_swap_atom_list;

  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Compute *c_pe;

  // arrays for dump image rendering

  int *imgobjs;
  double **imgparms;
  // maps atom IDs to number of steps they have been highlighted
  std::unordered_map<tagint, std::pair<int,int>> vizatoms;
  int vizsteps;                    // number of steps to highlight atoms in reactions

  void options(int, char **);
  double build_eatom_cache();
  int attempt_semi_grand();
  int attempt_swap();
  double energy_full();
  int pick_semi_grand_atom();
  int pick_i_swap_atom();
  int pick_j_swap_atom();
  void update_semi_grand_atoms_list();
  void update_swap_atoms_list();
};

}    // namespace LAMMPS_NS

#endif
#endif
