/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Paul Crozier (SNL)
                         Alexander Stukowski
------------------------------------------------------------------------- */

#include "fix_atom_swap.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "graphics.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "pair_hybrid_scaled.h"
#include "pair_pace.h"
#include "input.h"
#include "random_park.h"
#include "region.h"
#include "suffix.h"
#include "update.h"
#include "variable.h"

#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAtomSwap::FixAtomSwap(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), region(nullptr), idregion(nullptr), type_list(nullptr), mu(nullptr),
    mu_var_flag(nullptr), mu_var_index(nullptr), mu_var_names(nullptr),
    temp_var_flag(0), temp_var_index(-1), temp_var_name(nullptr),
    qtype(nullptr), mtype(nullptr), sqrt_mass_ratio(nullptr), local_swap_iatom_list(nullptr),
    local_swap_jatom_list(nullptr), local_swap_atom_list(nullptr), random_equal(nullptr),
    random_unequal(nullptr), c_pe(nullptr), imgobjs(nullptr), imgparms(nullptr)
{
  if (narg < 10) utils::missing_cmd_args(FLERR, "fix atom/swap", error);

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 5;    // [0]=attempts [1]=successes [2]=X [3]=chi [4]=mu_adapt
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // no visualization without an atom map
  if (atom->map_style == Atom::MAP_NONE) {
    vizsteps = 0;
  } else {
    vizsteps = 1000;
  }

  // required args

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  ncycles = utils::inumeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);

  if (nevery <= 0) error->all(FLERR, 3, "Illegal fix atom/swap command nevery value");
  if (ncycles < 0) error->all(FLERR, 4, "Illegal fix atom/swap command ncycles value");
  if (seed <= 0) error->all(FLERR, 5, "Illegal fix atom/swap command random seed");

  if (strncmp(arg[6], "v_", 2) == 0) {
    // equal-style variable for temperature
    temp_var_flag = 1;
    temp_var_name = utils::strdup(arg[6] + 2);
    beta = 0.0;    // placeholder; evaluated in init() and pre_exchange()
  } else {
    double temperature = utils::numeric(FLERR, arg[6], false, lmp);
    if (temperature <= 0.0) error->all(FLERR, 6, "Illegal fix atom/swap command temperature value");
    beta = 1.0 / (force->boltz * temperature);
  }

  memory->create(type_list, atom->ntypes, "atom/swap:type_list");
  memory->create(mu, atom->ntypes + 1, "atom/swap:mu");
  for (int i = 0; i <= atom->ntypes; i++) mu[i] = 0.0;

  memory->create(mu_var_flag, atom->ntypes + 1, "atom/swap:mu_var_flag");
  memory->create(mu_var_index, atom->ntypes + 1, "atom/swap:mu_var_index");
  mu_var_names = new char *[atom->ntypes + 1];
  for (int i = 0; i <= atom->ntypes; i++) {
    mu_var_flag[i] = 0;
    mu_var_index[i] = -1;
    mu_var_names[i] = nullptr;
  }

  // adaptive stepping defaults (disabled)
  adapt_flag = 0;
  adapt_type = -1;
  adapt_tracked_index = 1;    // default: track second listed type (type_list[1])
  adapt_dX = 0.01;
  adapt_every = 10;
  adapt_dmu_max = 10.0;
  adapt_mu_lo = -DBL_MAX;
  adapt_mu_hi =  DBL_MAX;
  adapt_counter = 0;
  adapt_sum_N2 = 0.0;
  adapt_sum_N2sq = 0.0;
  adapt_x_current = 0.0;
  adapt_chi_current = 0.0;
  adapt_mu_current = 0.0;

  // default value for multi-swap count
  nswap_count = 1;
  noforce_flag = 0;
  local_energy_flag = 0;
  eatom_cached = nullptr;
  eatom_cached_nmax = 0;
  split_cache_flag = 0;
  invariant_substyle = -1;
  eatom_sA = nullptr;
  eatom_sB = nullptr;
  eatom_s_nmax = 0;

  // read options from end of input line

  options(narg - 7, &arg[7]);

  // random number generator, same for all procs

  random_equal = new RanPark(lmp, seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp, seed);

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  mc_active = 0;

  nswap_attempts = 0.0;
  nswap_successes = 0.0;

  atom_swap_nmax = 0;
  local_swap_atom_list = nullptr;
  local_swap_iatom_list = nullptr;
  local_swap_jatom_list = nullptr;

  // set comm size needed by this Fix

  if (atom->q_flag)
    comm_forward = 2;
  else
    comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixAtomSwap::~FixAtomSwap()
{
  memory->destroy(type_list);
  memory->destroy(mu);
  memory->destroy(mu_var_flag);
  memory->destroy(mu_var_index);
  if (mu_var_names) {
    for (int i = 0; i <= atom->ntypes; i++) delete[] mu_var_names[i];
    delete[] mu_var_names;
  }
  delete[] temp_var_name;
  memory->destroy(qtype);
  memory->destroy(mtype);
  memory->destroy(sqrt_mass_ratio);
  memory->destroy(local_swap_iatom_list);
  memory->destroy(local_swap_jatom_list);
  delete[] idregion;
  delete random_equal;
  delete random_unequal;
  memory->destroy(imgobjs);
  memory->destroy(imgparms);
  memory->destroy(eatom_cached);
  memory->destroy(eatom_sA);
  memory->destroy(eatom_sB);
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixAtomSwap::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR, "Illegal fix atom/swap command");

  ke_flag = 1;
  semi_grand_flag = 0;
  nswaptypes = 0;
  nmutypes = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      region = domain->get_region_by_id(arg[iarg + 1]);
      if (!region) error->all(FLERR, "Region {} for fix atom/swap does not exist", arg[iarg + 1]);
      idregion = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ke") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      ke_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "semi-grand") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      semi_grand_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "types") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        if (nswaptypes >= atom->ntypes) error->all(FLERR, "Illegal fix atom/swap command");
        type_list[nswaptypes] = utils::expand_type_int(FLERR, arg[iarg], Atom::ATOM, lmp);
        nswaptypes++;
        iarg++;
      }
    } else if (strcmp(arg[iarg], "mu") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg < narg) {
        // stop at next keyword (alphabetic, not starting with 'v_')
        if (isalpha(arg[iarg][0]) && !(arg[iarg][0] == 'v' && arg[iarg][1] == '_')) break;
        nmutypes++;
        if (nmutypes > atom->ntypes) error->all(FLERR, "Illegal fix atom/swap command");
        if (strncmp(arg[iarg], "v_", 2) == 0) {
          // equal-style variable: store name (strip the "v_" prefix)
          mu_var_flag[nmutypes] = 1;
          mu_var_names[nmutypes] = utils::strdup(arg[iarg] + 2);
          mu[nmutypes] = 0.0;    // placeholder; evaluated at runtime
        } else {
          mu[nmutypes] = utils::numeric(FLERR, arg[iarg], false, lmp);
        }
        iarg++;
      }
    } else if (strcmp(arg[iarg], "swap_count") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      nswap_count = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (nswap_count < 1)
        error->all(FLERR, "Illegal fix atom/swap command: swap_count must be >= 1");
      iarg += 2;
    } else if (strcmp(arg[iarg], "noforce") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      noforce_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "localE") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      local_energy_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "adapt") == 0) {
      // adapt dX K [maxdmu value] [tracked N] [mumin val] [mumax val]
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      adapt_flag = 1;
      adapt_dX = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      adapt_every = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
      if (adapt_dX <= 0.0 || adapt_dX >= 1.0)
        error->all(FLERR, "Illegal fix atom/swap adapt dX: must be in (0, 1)");
      if (adapt_every < 1)
        error->all(FLERR, "Illegal fix atom/swap adapt K: must be >= 1");
      iarg += 3;
      // optional sub-keywords: maxdmu, tracked, mumin, mumax
      while (iarg < narg) {
        if (strcmp(arg[iarg], "maxdmu") == 0) {
          if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
          adapt_dmu_max = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
          if (adapt_dmu_max <= 0.0)
            error->all(FLERR, "Illegal fix atom/swap adapt maxdmu: must be positive");
          iarg += 2;
        } else if (strcmp(arg[iarg], "tracked") == 0) {
          // tracked N: N is 1 or 2 (1-based index into the types list)
          if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
          int idx = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
          if (idx < 1 || idx > 2)
            error->all(FLERR, "Illegal fix atom/swap adapt tracked: must be 1 or 2");
          adapt_tracked_index = idx - 1;    // convert to 0-based
          iarg += 2;
        } else if (strcmp(arg[iarg], "mumin") == 0) {
          if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
          adapt_mu_lo = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
          iarg += 2;
        } else if (strcmp(arg[iarg], "mumax") == 0) {
          if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
          adapt_mu_hi = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
          iarg += 2;
        } else
          break;
      }
    } else
      error->all(FLERR, "Illegal fix atom/swap command");
  }
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0],"vizsteps") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify atom/swap", error);
    vizsteps = utils::inumeric(FLERR, arg[1], false, lmp);
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::init()
{
  if ((atom->mass != nullptr) && (atom->rmass != nullptr) && (comm->me == 0))
    error->warning(FLERR, "Fix atom/swap will use per-atom masses for velocity rescaling");

  c_pe = modify->get_compute_by_id("thermo_pe");

  int *type = atom->type;

  if (nswaptypes < 2)
    error->all(FLERR, Error::NOLASTLINE,
               "Must specify at least 2 atom types in fix atom/swap command");

  // resolve and validate variable-backed temperature
  if (temp_var_flag) {
    temp_var_index = input->variable->find(temp_var_name);
    if (temp_var_index < 0)
      error->all(FLERR, "Variable {} for fix atom/swap temperature does not exist", temp_var_name);
    if (!input->variable->equalstyle(temp_var_index))
      error->all(FLERR, "Variable {} for fix atom/swap temperature must be equal-style",
                 temp_var_name);
    // evaluate immediately so beta is valid before first pre_exchange
    double temperature = input->variable->compute_equal(temp_var_index);
    if (temperature <= 0.0)
      error->all(FLERR, "Fix atom/swap variable temperature must be positive");
    beta = 1.0 / (force->boltz * temperature);
  }

  if (semi_grand_flag) {
    if (nswaptypes != nmutypes)
      error->all(FLERR, Error::NOLASTLINE, "Need nswaptypes mu values in fix atom/swap command");
    // resolve and validate any variable-backed mu values
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      int itype = type_list[iswaptype];
      if (mu_var_flag[itype]) {
        mu_var_index[itype] = input->variable->find(mu_var_names[itype]);
        if (mu_var_index[itype] < 0)
          error->all(FLERR, "Variable {} for fix atom/swap mu does not exist",
                     mu_var_names[itype]);
        if (!input->variable->equalstyle(mu_var_index[itype]))
          error->all(FLERR, "Variable {} for fix atom/swap mu must be equal-style",
                     mu_var_names[itype]);
      }
    }
    // validate adaptive stepping
    if (adapt_flag) {
      if (mu_var_flag[type_list[1]])
        error->all(FLERR, Error::NOLASTLINE,
                   "Fix atom/swap adapt is not compatible with v_ variable mu");
      if (nswaptypes != 2)
        error->all(FLERR, Error::NOLASTLINE,
                   "Fix atom/swap adapt requires exactly 2 swap types");
      adapt_type = type_list[adapt_tracked_index];   // user-selected or default type_list[1]
      adapt_mu_current = mu[adapt_type];
      if (adapt_mu_lo >= adapt_mu_hi)
        error->all(FLERR, "Fix atom/swap adapt: mumin must be less than mumax");
    }
  } else {
    if (nswaptypes != 2)
      error->all(FLERR, Error::NOLASTLINE,
                 "Exactly 2 atom types must be used without semi-grand keyword in fix atom/swap");
    if (nmutypes != 0)
      error->all(FLERR, Error::NOLASTLINE,
                 "Mu not allowed when not using semi-grand in fix atom/swap command");
  }

  // must have a pair style and not use INTEL package

  if (!force->pair) error->all(FLERR, Error::NOLASTLINE, "Fix atom/swap requires a pair style");
  if (force->pair && (force->pair->suffix_flag & Suffix::INTEL))
    error->all(FLERR, Error::NOLASTLINE, "Fix {} is not compatible with /intel pair styles", style);

  // check if constraints for hybrid pair styles are fulfilled

  if (utils::strmatch(force->pair_style, "^hybrid")) {
    auto *hybrid = dynamic_cast<PairHybrid *>(force->pair);
    if (hybrid) {
      for (int i = 0; i < nswaptypes - 1; ++i) {
        int type1 = type_list[i];
        for (int j = i + 1; j < nswaptypes; ++j) {
          int type2 = type_list[j];
          if (hybrid->nmap[type1][type1] != hybrid->nmap[type2][type2])
            error->all(FLERR, Error::NOLASTLINE,
                       "Pair {} substyles for atom types {} and {} are not compatible",
                       force->pair_style, type1, type2);
          for (int k = 0; k < hybrid->nmap[type1][type1]; ++k) {
            if (hybrid->map[type1][type1][k] != hybrid->map[type2][type2][k])
              error->all(FLERR, Error::NOLASTLINE,
                         "Pair {} substyles for atom types {} and {} are not compatible",
                         force->pair_style, type1, type2);
          }
        }
      }
    }
  }

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region)
      error->all(FLERR, Error::NOLASTLINE, "Region {} for fix atom/swap does not exist", idregion);
  }

  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    if (type_list[iswaptype] <= 0 || type_list[iswaptype] > atom->ntypes)
      error->all(FLERR, "Invalid atom type in fix atom/swap command");

  // this is only required for non-semi-grand
  // in which case, nswaptypes = 2

  if (atom->q_flag && !semi_grand_flag) {
    double qmax, qmin;
    int firstall, first;
    memory->create(qtype, nswaptypes, "atom/swap:qtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first) {
              qtype[iswaptype] = atom->q[i];
              first = 0;
            } else if (qtype[iswaptype] != atom->q[i])
              error->one(FLERR, "All atoms of a swapped type must have the same charge.");
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall)
        error->all(FLERR,
                   "At least one atom of each swapped type must be present to define charges.");
      if (first) qtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) qtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (qmax != qmin) error->all(FLERR, "All atoms of a swapped type must have same charge.");
      qtype[iswaptype] = qmax;
    }
  }

  // if we have per-atom masses, check that rmass is consistent with type,
  // and set per-type mass to that value
  if ((atom->rmass !=  nullptr) && !semi_grand_flag) {
    double mmax, mmin;
    int firstall, first;
    memory->create(mtype, nswaptypes, "atom/swap:mtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first > 0) {
              mtype[iswaptype] = atom->rmass[i];
              first = 0;
            } else if (mtype[iswaptype] != atom->rmass[i])
              first = -1;
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall < 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "All atoms of a swapped type must have the same per-atom mass");
      if (firstall > 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "At least one atom of each swapped type must be present to define masses");
      if (first) mtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) mtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (mmax != mmin)
        error->all(FLERR, Error::NOLASTLINE, "All atoms of a swapped type must have same mass.");
      mtype[iswaptype] = mmax;
    }
  }

  memory->create(sqrt_mass_ratio, atom->ntypes + 1, atom->ntypes + 1, "atom/swap:sqrt_mass_ratio");
  if (atom->rmass != nullptr) {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) sqrt_mass_ratio[itype][jtype] = 1.0;
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      int itype = type_list[iswaptype];
      for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++) {
        int jtype = type_list[jswaptype];
        sqrt_mass_ratio[itype][jtype] = sqrt(mtype[iswaptype] / mtype[jswaptype]);
      }
    }
  } else {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++)
        sqrt_mass_ratio[itype][jtype] = sqrt(atom->mass[itype] / atom->mass[jtype]);
  }

  // check to see if itype and jtype cutoffs are the same
  // if not, reneighboring will be needed between swaps

  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      for (int ktype = 1; ktype <= atom->ntypes; ktype++)
        if (cutsq[type_list[iswaptype]][ktype] != cutsq[type_list[jswaptype]][ktype])
          unequal_cutoffs = true;

  // localE validation: PACE (plain or hybrid/scaled all-PACE), single-swap, equal cutoffs
  if (local_energy_flag) {
    if (nswap_count > 1)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap localE is not compatible with swap_count > 1");
    if (unequal_cutoffs)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap localE is not compatible with unequal type cutoffs");

    // populate pace_substyles: allows plain 'pace' or 'hybrid/scaled' with all-PACE sub-styles
    // for non-PACE pair styles (e.g. GRACE), localE is silently ignored
    pace_substyles.clear();
    bool is_pace_style = false;
    auto *hybrid_sc = dynamic_cast<PairHybridScaled *>(force->pair);
    if (hybrid_sc) {
      bool all_pace = true;
      for (int s = 0; s < hybrid_sc->nstyles; s++) {
        auto *pace_s = dynamic_cast<PairPACE *>(hybrid_sc->styles[s]);
        if (!pace_s) { all_pace = false; break; }
        pace_substyles.emplace_back(pace_s, hybrid_sc->scaleval[s]);
      }
      is_pace_style = all_pace;
      if (!is_pace_style) pace_substyles.clear();
    } else {
      auto *pace = dynamic_cast<PairPACE *>(force->pair);
      if (pace) {
        pace_substyles.emplace_back(pace, 1.0);
        is_pace_style = true;
      }
    }

    if (!is_pace_style) {
      // non-PACE pair style (e.g. GRACE): silently disable localE optimization
      if (comm->me == 0)
        utils::logmesg(lmp, "Fix atom/swap localE: pair style is not pace, "
                            "localE optimization disabled (no-op)\n");
      local_energy_flag = 0;
    }

    // semi-grand localE supports only the single plain-PACE fast path;
    // the hybrid/scaled split-cache machinery is specific to (non-semi-grand)
    // alchemical TI runs.
    if (local_energy_flag && semi_grand_flag &&
        !(pace_substyles.size() == 1 && pace_substyles[0].second == 1.0))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap localE with semi-grand requires a single "
                 "pair_style pace (no hybrid/scaled)");

    // pre-allocate the per-atom energy cache
    if (atom->nlocal > eatom_cached_nmax) {
      memory->destroy(eatom_cached);
      eatom_cached_nmax = atom->nlocal + 100;
      memory->create(eatom_cached, eatom_cached_nmax, "atom/swap:eatom_cached");
    }

    // detect type-invariant sub-style (e.g. "Au Au" endpoint in alchemical TI)
    split_cache_flag = 0;
    invariant_substyle = -1;
    if (pace_substyles.size() == 2) {
      for (int s = 0; s < 2; s++) {
        if (pace_substyles[s].first->is_type_invariant(type_list[0], type_list[1])) {
          invariant_substyle = s;
          split_cache_flag = 1;
          break;
        }
      }
    }
    // allocate per-style unscaled cache arrays for the split-cache path
    if (split_cache_flag) {
      if (atom->nlocal > eatom_s_nmax) {
        memory->destroy(eatom_sA);
        memory->destroy(eatom_sB);
        eatom_s_nmax = atom->nlocal + 100;
        memory->create(eatom_sA, eatom_s_nmax, "atom/swap:eatom_sA");
        memory->create(eatom_sB, eatom_s_nmax, "atom/swap:eatom_sB");
      }
    }
  }

  // check that no swappable atoms are in atom->firstgroup
  // swapping such an atom might not leave firstgroup atoms first

  if (atom->firstgroup >= 0) {
    int *mask = atom->mask;
    int firstgroupbit = group->bitmask[atom->firstgroup];

    int flag = 0;
    for (int i = 0; i < atom->nlocal; i++)
      if ((mask[i] == groupbit) && (mask[i] && firstgroupbit)) flag = 1;

    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);

    if (flagall) error->all(FLERR, "Cannot do atom/swap on atoms in atom_modify first group");
  }
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo swaps
------------------------------------------------------------------------- */

void FixAtomSwap::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  mc_active = 1;

  // ensure current system is ready to compute energy

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  // energy_stored = energy of current state
  // will be updated after accepted swaps

  // energy_stored = energy of current state (full or cached per-atom sum)
  if (local_energy_flag)
    energy_stored = build_eatom_cache();
  else
    energy_stored = energy_full();

  // evaluate variable-backed temperature and mu values once per MC block
  if (temp_var_flag) {
    double temperature = input->variable->compute_equal(temp_var_index);
    if (temperature <= 0.0)
      error->all(FLERR, "Fix atom/swap variable temperature must be positive");
    beta = 1.0 / (force->boltz * temperature);
  }
  if (semi_grand_flag) {
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      int itype = type_list[iswaptype];
      if (mu_var_flag[itype])
        mu[itype] = input->variable->compute_equal(mu_var_index[itype]);
    }
  }

  // attempt Ncycle atom swaps

  int nsuccess = 0;
  if (semi_grand_flag) {
    update_semi_grand_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
  } else {
    update_swap_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap();
  }

  // adaptive susceptibility-driven mu stepping (semi-grand, 2-type only)
  if (adapt_flag) {
    // count N_adapt_type globally after this block's accepted swaps
    int n2_local = 0;
    int *type = atom->type;
    for (int i = 0; i < atom->nlocal; i++)
      if (type[i] == adapt_type) n2_local++;
    int n2_global = 0;
    MPI_Allreduce(&n2_local, &n2_global, 1, MPI_INT, MPI_SUM, world);

    int n_total = atom->natoms;
    adapt_x_current = static_cast<double>(n2_global) / static_cast<double>(n_total);

    adapt_sum_N2   += static_cast<double>(n2_global);
    adapt_sum_N2sq += static_cast<double>(n2_global) * static_cast<double>(n2_global);
    adapt_counter++;

    if (adapt_counter >= adapt_every) {
      double mean_N2  = adapt_sum_N2   / adapt_counter;
      double mean_N2sq = adapt_sum_N2sq / adapt_counter;
      double var_N2   = mean_N2sq - mean_N2 * mean_N2;

      // chi = beta * Var(N2) / N_total   [dimensionless dX/dmu]
      double chi = beta * var_N2 / static_cast<double>(n_total);
      adapt_chi_current = chi;

      // step mu; clamp to maxdmu to avoid runaway in flat regions
      double dmu = 0.0;
      if (chi > 1.0e-10)
        dmu = adapt_dX / chi;
      else
        dmu = adapt_dmu_max;    // flat region: take max step

      if (dmu >  adapt_dmu_max) dmu =  adapt_dmu_max;
      if (dmu < -adapt_dmu_max) dmu = -adapt_dmu_max;

      mu[adapt_type] += dmu;
      if (mu[adapt_type] < adapt_mu_lo) mu[adapt_type] = adapt_mu_lo;
      if (mu[adapt_type] > adapt_mu_hi) mu[adapt_type] = adapt_mu_hi;
      adapt_mu_current = mu[adapt_type];

      // reset accumulators
      adapt_sum_N2   = 0.0;
      adapt_sum_N2sq = 0.0;
      adapt_counter  = 0;
    }
  }

  // update MC stats

  nswap_attempts += ncycles;
  nswap_successes += nsuccess;

  next_reneighbor = update->ntimestep + nevery;

  mc_active = 0;

  // if visualization support is enabled, age vizatoms and remove expired ones
  if (vizsteps > 0) {
    std::vector<tagint> eraseme;
    for (const auto &[key, data] : vizatoms) {
      int idx = atom->map(key);
      if ((idx < 0) || (data.first < 0)) {
        eraseme.push_back(key);
        continue;
      }
      vizatoms[key] = std::make_pair(data.first - nevery, data.second);
    }
    for (const auto &key : eraseme) vizatoms.erase(key);
  }
}

/* ----------------------------------------------------------------------
   attempt a semd-grand swap of a single atom
   compare before/after energy and accept/reject the swap
   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_semi_grand()
{
  if (nswap == 0) return 0;

  // pre-swap energy

  double energy_before = energy_stored;

  // pick a random atom and perform swap

  int itype = -1, jtype = -1, jswaptype;
  int i = pick_semi_grand_atom();
  if (i >= 0) {
    jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
    jtype = type_list[jswaptype];
    itype = atom->type[i];
    while (itype == jtype) {
      jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
      jtype = type_list[jswaptype];
    }
    atom->type[i] = jtype;
  }

  // for localE: broadcast the global tag of the selected atom to all ranks
  tagint tag_i_global = 0;
  if (local_energy_flag) {
    tagint tag_send = (i >= 0) ? atom->tag[i] : (tagint) 0;
    MPI_Allreduce(&tag_send, &tag_i_global, 1, MPI_LMP_TAGINT, MPI_SUM, world);
  }

  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  // post-swap energy: local shell approximation (localE) or full pair compute
  // for localE, semi-grand uses the single plain-PACE fast path (enforced in
  // init), with the single selected atom passed as tag_i and tag_j = 0

  double energy_after;
  std::vector<std::pair<int, double>> changed_atoms;

  if (local_energy_flag) {
    double local_dE = pace_substyles[0].first->compute_shell_delta(
        tag_i_global, (tagint) 0, eatom_cached, changed_atoms);
    double total_dE;
    MPI_Allreduce(&local_dE, &total_dE, 1, MPI_DOUBLE, MPI_SUM, world);
    energy_after = energy_stored + total_dE;
  } else {
    if (force->kspace) force->kspace->qsum_qsq();
    energy_after = energy_full();
  }

  int success = 0;
  if (i >= 0)
    if (random_unequal->uniform() <
        exp(beta * (energy_before - energy_after + mu[jtype] - mu[itype])))
      success = 1;

  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  // swap accepted, return 1

  if (success_all) {
    update_semi_grand_atoms_list();
    // update per-atom energy cache with recomputed shell energies
    if (local_energy_flag)
      for (auto &[idx, new_e] : changed_atoms) eatom_cached[idx] = new_e;
    energy_stored = energy_after;
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
        // record atom for which the type was swapped and store the old type
        if (vizsteps > 0) {
          vizatoms[atom->tag[i]] = std::make_pair(vizsteps,itype);
        }
      }
    }
    return 1;
  }

  // swap not accepted, return 0
  // restore the swapped atom
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes

  if (i >= 0) atom->type[i] = itype;
  if (force->kspace) force->kspace->qsum_qsq();

  // re-sync ghost types after rejected swap so next trial sees correct types
  if (local_energy_flag && !unequal_cutoffs) comm->forward_comm(this);

  return 0;
}

/* ----------------------------------------------------------------------
   attempt a swap of one or more pairs of atoms
   nswap_count pairs are selected and swapped atomically as a single MC move
   compare before/after total energy and accept/reject via Metropolis criterion
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_swap()
{
  if ((niswap < nswap_count) || (njswap < nswap_count)) return 0;

  // pre-swap energy

  double energy_before = energy_stored;

  int itype = type_list[0];
  int jtype = type_list[1];

  // pick nswap_count distinct i-type and j-type atoms
  // local index is -1 when the atom resides on another proc

  std::vector<int> i_picks(nswap_count, -1);
  std::vector<int> j_picks(nswap_count, -1);
  std::vector<int> i_global_picks;
  std::vector<int> j_global_picks;
  i_global_picks.reserve(nswap_count);
  j_global_picks.reserve(nswap_count);

  for (int k = 0; k < nswap_count; k++) {
    int iwhichglobal;
    bool is_dup;
    do {
      iwhichglobal = static_cast<int>(niswap * random_equal->uniform());
      is_dup = false;
      for (int prev : i_global_picks)
        if (prev == iwhichglobal) {
          is_dup = true;
          break;
        }
    } while (is_dup);
    i_global_picks.push_back(iwhichglobal);
    if ((iwhichglobal >= niswap_before) && (iwhichglobal < niswap_before + niswap_local))
      i_picks[k] = local_swap_iatom_list[iwhichglobal - niswap_before];
  }

  for (int k = 0; k < nswap_count; k++) {
    int jwhichglobal;
    bool is_dup;
    do {
      jwhichglobal = static_cast<int>(njswap * random_equal->uniform());
      is_dup = false;
      for (int prev : j_global_picks)
        if (prev == jwhichglobal) {
          is_dup = true;
          break;
        }
    } while (is_dup);
    j_global_picks.push_back(jwhichglobal);
    if ((jwhichglobal >= njswap_before) && (jwhichglobal < njswap_before + njswap_local))
      j_picks[k] = local_swap_jatom_list[jwhichglobal - njswap_before];
  }

  // for localE: broadcast the global atom tags of the selected pair to all ranks
  tagint tag_i_global = 0, tag_j_global = 0;
  if (local_energy_flag) {
    tagint tags_send[2] = {(i_picks[0] >= 0) ? atom->tag[i_picks[0]] : (tagint) 0,
                           (j_picks[0] >= 0) ? atom->tag[j_picks[0]] : (tagint) 0};
    tagint tags_recv[2] = {0, 0};
    MPI_Allreduce(tags_send, tags_recv, 2, MPI_LMP_TAGINT, MPI_SUM, world);
    tag_i_global = tags_recv[0];
    tag_j_global = tags_recv[1];
  }

  // swap types (and charges/masses) of all selected atoms

  for (int k = 0; k < nswap_count; k++) {
    int i = i_picks[k];
    if (i >= 0) {
      atom->type[i] = jtype;
      if (atom->q_flag) atom->q[i] = qtype[1];
      if (atom->rmass != nullptr) atom->rmass[i] = mtype[1];
    }
  }
  for (int k = 0; k < nswap_count; k++) {
    int j = j_picks[k];
    if (j >= 0) {
      atom->type[j] = itype;
      if (atom->q_flag) atom->q[j] = qtype[0];
      if (atom->rmass != nullptr) atom->rmass[j] = mtype[0];
    }
  }

  // if unequal_cutoffs, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

  if (unequal_cutoffs) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  // post-swap energy: local shell approximation (localE) or full pair compute

  double energy_after;
  std::vector<std::pair<int, double>> changed_atoms;
  std::vector<double> changed_eB;  // new type-aware energies (split-cache path only)

  if (local_energy_flag) {
    double local_dE;
    if (pace_substyles.size() == 1 && pace_substyles[0].second == 1.0) {
      // fast path: single PACE, use compute_shell_delta directly
      local_dE = pace_substyles[0].first->compute_shell_delta(tag_i_global, tag_j_global,
                                                               eatom_cached, changed_atoms);
    } else {
      // hybrid/scaled path: find affected atoms via first sub-style's neighbour list
      // (equal cutoffs are enforced in init()), then sum scaled contributions
      std::vector<int> affected;
      pace_substyles[0].first->get_affected_local_atoms(tag_i_global, tag_j_global, affected);
      local_dE = 0.0;
      if (split_cache_flag) {
        // type-invariant sub-style energy is unchanged by a type swap; reuse cached value
        int type_aware = 1 - invariant_substyle;
        double sc_inv   = pace_substyles[invariant_substyle].second;
        double sc_aware = pace_substyles[type_aware].second;
        for (int k : affected) {
          double new_eB = pace_substyles[type_aware].first->compute_atom_energy(k);
          double new_e  = sc_inv * eatom_sA[k] + sc_aware * new_eB;
          local_dE += new_e - eatom_cached[k];
          changed_atoms.emplace_back(k, new_e);
          changed_eB.push_back(new_eB);
        }
      } else {
        for (int k : affected) {
          double new_e = 0.0;
          for (auto &[pace_s, scale_s] : pace_substyles)
            new_e += scale_s * pace_s->compute_atom_energy(k);
          local_dE += new_e - eatom_cached[k];
          changed_atoms.emplace_back(k, new_e);
        }
      }
    }
    double total_dE;
    MPI_Allreduce(&local_dE, &total_dE, 1, MPI_DOUBLE, MPI_SUM, world);
    energy_after = energy_stored + total_dE;
  } else {
    energy_after = energy_full();
  }

  // swap accepted, return 1
  // if ke_flag, rescale atom velocities

  if (random_equal->uniform() < exp(beta * (energy_before - energy_after))) {
    update_swap_atoms_list();
    if (ke_flag) {
      for (int k = 0; k < nswap_count; k++) {
        int i = i_picks[k];
        if (i >= 0) {
          atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
          atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
          atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
        }
        int j = j_picks[k];
        if (j >= 0) {
          atom->v[j][0] *= sqrt_mass_ratio[jtype][itype];
          atom->v[j][1] *= sqrt_mass_ratio[jtype][itype];
          atom->v[j][2] *= sqrt_mass_ratio[jtype][itype];
        }
      }
      // record atoms for which the type was swapped and store the old types
      if (vizsteps > 0) {
        for (int k = 0; k < nswap_count; k++) {
          int i = i_picks[k];
          int j = j_picks[k];
          if (i >= 0) vizatoms[atom->tag[i]] = std::make_pair(vizsteps, jtype);
          if (j >= 0) vizatoms[atom->tag[j]] = std::make_pair(vizsteps, itype);
        }
      }
    }
    // update per-atom energy cache with recomputed shell energies
    if (local_energy_flag) {
      for (auto &[idx, new_e] : changed_atoms) eatom_cached[idx] = new_e;
      // split-cache: persist new type-aware energies; invariant sub-style (eatom_sA) is unchanged
      if (split_cache_flag)
        for (size_t ci = 0; ci < changed_atoms.size(); ci++)
          eatom_sB[changed_atoms[ci].first] = changed_eB[ci];
    }
    energy_stored = energy_after;
    return 1;
  }

  // swap not accepted, return 0
  // restore the swapped itype & jtype atoms
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes

  for (int k = 0; k < nswap_count; k++) {
    int i = i_picks[k];
    if (i >= 0) {
      atom->type[i] = type_list[0];
      if (atom->q_flag) atom->q[i] = qtype[0];
      if (atom->rmass != nullptr) atom->rmass[i] = mtype[0];
    }
    int j = j_picks[k];
    if (j >= 0) {
      atom->type[j] = type_list[1];
      if (atom->q_flag) atom->q[j] = qtype[1];
      if (atom->rmass != nullptr) atom->rmass[j] = mtype[1];
    }
  }

  // re-sync ghost types after rejected swap so next trial sees correct types
  if (local_energy_flag && !unequal_cutoffs) comm->forward_comm(this);

  return 0;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixAtomSwap::energy_full()
{
  int eflag = 1;
  int vflag = 0;

  if (modify->n_pre_force) modify->pre_force(vflag);

  if (noforce_flag && force->pair) {
    auto *hybrid = dynamic_cast<PairHybrid *>(force->pair);
    if (hybrid) {
      for (int s = 0; s < hybrid->nstyles; s++) hybrid->styles[s]->energy_only = 1;
    } else {
      force->pair->energy_only = 1;
    }
  }
  if (force->pair) force->pair->compute(eflag, vflag);
  if (noforce_flag && force->pair) {
    auto *hybrid = dynamic_cast<PairHybrid *>(force->pair);
    if (hybrid) {
      for (int s = 0; s < hybrid->nstyles; s++) hybrid->styles[s]->energy_only = 0;
    } else {
      force->pair->energy_only = 0;
    }
  }

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  if (modify->n_post_force_any) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  return c_pe->compute_scalar();
}

/* ----------------------------------------------------------------------
   build the per-atom ACE energy cache after reneighboring.
   returns the global total PE (sum of all e_atom over all ranks).
------------------------------------------------------------------------- */

double FixAtomSwap::build_eatom_cache()
{
  // grow array if nlocal has increased since last allocation
  int nlocal = atom->nlocal;
  if (nlocal > eatom_cached_nmax) {
    memory->destroy(eatom_cached);
    eatom_cached_nmax = nlocal + 100;
    memory->create(eatom_cached, eatom_cached_nmax, "atom/swap:eatom_cached");
  }

  // refresh scale factors in case hybrid/scaled uses variable-based scales
  auto *hybrid_sc = dynamic_cast<PairHybridScaled *>(force->pair);
  if (hybrid_sc)
    for (size_t s = 0; s < pace_substyles.size(); s++)
      pace_substyles[s].second = hybrid_sc->scaleval[s];

  double local_sum;
  if (pace_substyles.size() == 1 && pace_substyles[0].second == 1.0) {
    // fast path: single PACE sub-style with unit scale
    local_sum = pace_substyles[0].first->build_atom_energy_cache(eatom_cached, eatom_cached_nmax);
  } else if (split_cache_flag) {
    // split-cache path: keep per-style unscaled caches for trial optimization
    int type_aware = 1 - invariant_substyle;
    if (nlocal > eatom_s_nmax) {
      memory->destroy(eatom_sA);
      memory->destroy(eatom_sB);
      eatom_s_nmax = nlocal + 100;
      memory->create(eatom_sA, eatom_s_nmax, "atom/swap:eatom_sA");
      memory->create(eatom_sB, eatom_s_nmax, "atom/swap:eatom_sB");
    }
    for (int i = 0; i < nlocal; i++) { eatom_sA[i] = 0.0; eatom_sB[i] = 0.0; }
    pace_substyles[invariant_substyle].first->accumulate_atom_energies(1.0, eatom_sA, eatom_s_nmax);
    pace_substyles[type_aware].first->accumulate_atom_energies(1.0, eatom_sB, eatom_s_nmax);
    double sc_inv   = pace_substyles[invariant_substyle].second;
    double sc_aware = pace_substyles[type_aware].second;
    local_sum = 0.0;
    for (int i = 0; i < nlocal; i++) {
      eatom_cached[i] = sc_inv * eatom_sA[i] + sc_aware * eatom_sB[i];
      local_sum += eatom_cached[i];
    }
  } else {
    // hybrid/scaled path: zero cache then accumulate each scaled sub-style
    for (int i = 0; i < nlocal; i++) eatom_cached[i] = 0.0;
    for (auto &[pace_s, scale_s] : pace_substyles)
      pace_s->accumulate_atom_energies(scale_s, eatom_cached, eatom_cached_nmax);
    local_sum = 0.0;
    for (int i = 0; i < nlocal; i++) local_sum += eatom_cached[i];
  }

  double global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, world);
  return global_sum;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_semi_grand_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int>(nswap * random_equal->uniform());
  if ((iwhichglobal >= nswap_before) && (iwhichglobal < nswap_before + nswap_local)) {
    int iwhichlocal = iwhichglobal - nswap_before;
    i = local_swap_atom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_i_swap_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int>(niswap * random_equal->uniform());
  if ((iwhichglobal >= niswap_before) && (iwhichglobal < niswap_before + niswap_local)) {
    int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_j_swap_atom()
{
  int j = -1;
  int jwhichglobal = static_cast<int>(njswap * random_equal->uniform());
  if ((jwhichglobal >= njswap_before) && (jwhichglobal < njswap_before + njswap_local)) {
    int jwhichlocal = jwhichglobal - njswap_before;
    j = local_swap_jatom_list[jwhichlocal];
  }

  return j;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_semi_grand_atoms_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_atom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_atom_list, atom_swap_nmax, "MCSWAP:local_swap_atom_list");
  }

  nswap_local = 0;

  if (region) {
    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          int itype = atom->type[i];
          int iswaptype;
          for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
            if (itype == type_list[iswaptype]) break;
          if (iswaptype == nswaptypes) continue;
          local_swap_atom_list[nswap_local] = i;
          nswap_local++;
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        int itype = atom->type[i];
        int iswaptype;
        for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
          if (itype == type_list[iswaptype]) break;
        if (iswaptype == nswaptypes) continue;
        local_swap_atom_list[nswap_local] = i;
        nswap_local++;
      }
    }
  }

  MPI_Allreduce(&nswap_local, &nswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&nswap_local, &nswap_before, 1, MPI_INT, MPI_SUM, world);
  nswap_before -= nswap_local;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_swap_atoms_list()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_iatom_list);
    memory->destroy(local_swap_jatom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_iatom_list, atom_swap_nmax, "MCSWAP:local_swap_iatom_list");
    memory->create(local_swap_jatom_list, atom_swap_nmax, "MCSWAP:local_swap_jatom_list");
  }

  niswap_local = 0;
  njswap_local = 0;

  if (region) {

    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[0]) {
            local_swap_iatom_list[niswap_local] = i;
            niswap_local++;
          } else if (type[i] == type_list[1]) {
            local_swap_jatom_list[njswap_local] = i;
            njswap_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        if (type[i] == type_list[0]) {
          local_swap_iatom_list[niswap_local] = i;
          niswap_local++;
        } else if (type[i] == type_list[1]) {
          local_swap_jatom_list[njswap_local] = i;
          njswap_local++;
        }
      }
    }
  }

  MPI_Allreduce(&niswap_local, &niswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&niswap_local, &niswap_before, 1, MPI_INT, MPI_SUM, world);
  niswap_before -= niswap_local;

  MPI_Allreduce(&njswap_local, &njswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&njswap_local, &njswap_before, 1, MPI_INT, MPI_SUM, world);
  njswap_before -= njswap_local;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;

  if (atom->q_flag) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
      buf[m++] = q[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;
  last = first + n;

  if (atom->q_flag) {
    for (i = first; i < last; i++) {
      type[i] = static_cast<int>(buf[m++]);
      q[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++) type[i] = static_cast<int>(buf[m++]);
  }
}

/* ----------------------------------------------------------------------
  return acceptance ratio
------------------------------------------------------------------------- */

double FixAtomSwap::compute_vector(int n)
{
  if (n == 0) return nswap_attempts;
  if (n == 1) return nswap_successes;
  if (n == 2) return adapt_x_current;
  if (n == 3) return adapt_chi_current;
  if (n == 4) return adapt_mu_current;
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixAtomSwap::memory_usage()
{
  double bytes = (double) atom_swap_nmax * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixAtomSwap::write_restart(FILE *fp)
{
  int n = 0;
  double list[8];
  list[n++] = random_equal->state();
  list[n++] = random_unequal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = nswap_attempts;
  list[n++] = nswap_successes;
  list[n++] = ubuf(update->ntimestep).d;
  list[n++] = adapt_mu_current;   // adaptive mu value (valid even when adapt_flag=0)
  list[n++] = ubuf(adapt_counter).d;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list, sizeof(double), n, fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixAtomSwap::restart(char *buf)
{
  int n = 0;
  auto *list = (double *) buf;

  seed = static_cast<int>(list[n++]);
  random_equal->reset(seed);

  seed = static_cast<int>(list[n++]);
  random_unequal->reset(seed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  nswap_attempts = static_cast<int>(list[n++]);
  nswap_successes = static_cast<int>(list[n++]);

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR, "Must not reset timestep when restarting fix atom/swap");

  // restore adaptive mu (always written; only applied when adapt_flag is on)
  double saved_mu = list[n++];
  adapt_counter = static_cast<int>(ubuf(list[n++]).i);
  if (adapt_flag) {
    mu[adapt_type] = saved_mu;
    adapt_mu_current = saved_mu;
  }
  // reset partial accumulators so the resumed run starts clean
  adapt_sum_N2   = 0.0;
  adapt_sum_N2sq = 0.0;
}

/* ----------------------------------------------------------------------
   extract variable which stores whether MC is active or not
     active = MC moves are taking place
     not active = normal MD is taking place
------------------------------------------------------------------------- */

void *FixAtomSwap::extract(const char *name, int &dim)
{
  if (strcmp(name,"mc_active") == 0) {
    dim = 0;
    return (void *) &mc_active;
  }
  return nullptr;
}

/* ----------------------------------------------------------------------
   provide graphics information to dump image to render spheres
   at the location of atoms that were involved in a reaction
------------------------------------------------------------------------- */

int FixAtomSwap::image(int *&objs, double **&parms)
{
  // no visualization without an atom map
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, Error::NOLASTLINE,
               "Cannot use fix atom/swap in dump image without an atom map");

  memory->destroy(imgobjs);
  memory->destroy(imgparms);

  int numobjs = vizatoms.size();
  int n = 0;
  if (numobjs > 0) {
    memory->create(imgobjs, numobjs, "atom/swap:imgobjs");
    memory->create(imgparms, numobjs, 5, "atom/swap:imgparms");

    int idx;
    const auto *const *const x = atom->x;
    for (const auto &[key, data] : vizatoms) {
      idx = atom->map(key);
      if (idx < 0) continue;
      imgobjs[n] = Graphics::SPHERE;
      imgparms[n][0] = data.second; // use stored pre-swap atom type
      imgparms[n][1] = x[idx][0];
      imgparms[n][2] = x[idx][1];
      imgparms[n][3] = x[idx][2];
      imgparms[n][4] = 0.0;     // radius is set with fflag2 in dump image
      ++n;
    }
  }
  objs = imgobjs;
  parms = imgparms;
  return n;
}
