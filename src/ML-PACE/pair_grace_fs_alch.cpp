/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Alchemical GRACE/FS pair style — a copy of pair_grace_fs.cpp driving
// GRACEFSAlchEvaluator instead of GRACEFSBEvaluator, with the per-atom
// lambda plumbing added (marked ALCHEMY). Extrapolation-grade and
// energy-only machinery stripped for legibility.
//

#include "pair_grace_fs_alch.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cstring>
#include <exception>

#include "ace/grace_fs_alch_evaluator.h"
#include "utils_pace.h"

namespace LAMMPS_NS {
struct ACEAlchImpl {
  ACEAlchImpl() : basis_set(nullptr), ace(nullptr) {}

  ~ACEAlchImpl()
  {
    delete basis_set;
    delete ace;
  }

  GRACEFSBasisSet *basis_set;
  GRACEFSAlchEvaluator *ace;
};
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
PairGRACEFSAlch::PairGRACEFSAlch(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  aceimpl = new ACEAlchImpl;
  scale = nullptr;

  comm_reverse = 1;    // ALCHEMY: dedlam contributions on ghost atoms

  centroidstressflag = CENTROID_AVAIL;
}

/* ---------------------------------------------------------------------- */

PairGRACEFSAlch::~PairGRACEFSAlch()
{
  if (copymode) return;

  delete aceimpl;
  memory->destroy(dedlam);
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairGRACEFSAlch::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum;
  double delx, dely, delz, evdwl;
  double fij[3];
  int *ilist, *jlist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  inum = list->inum;
  if (inum == 0) return;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // ALCHEMY: per-atom lambda (locals + ghosts, fix property/atom ghost yes)
  aceimpl->ace->lambda = atom->dvector[index_lambda];

  // ALCHEMY: per-atom dE/dlambda accumulator over locals + ghosts
  if (atom->nmax > nmax_dedlam) {
    memory->destroy(dedlam);
    nmax_dedlam = atom->nmax;
    memory->create(dedlam, nmax_dedlam, "pair:dedlam");
  }
  const int nall = atom->nlocal + atom->nghost;
  for (i = 0; i < nall; i++) dedlam[i] = 0.0;

  //determine the maximum number of neighbours
  int max_jnum = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    if (jnum > max_jnum) max_jnum = jnum;
  }

  aceimpl->ace->resize_neighbours_cache(max_jnum);
  std::vector<int> my_neigh_jlist(max_jnum);

  //loop over atoms
  for (ii = 0; ii < inum; ii++) {
    i = list->ilist[ii];
    const int itype = type[i];

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    // apply NEIGHMASK
    for (jj = 0; jj < jnum; ++jj) { my_neigh_jlist[jj] = jlist[jj] & NEIGHMASK; }
    try {
      aceimpl->ace->compute_atom(i, x, type, jnum, my_neigh_jlist.data());
    } catch (std::exception &e) {
      error->one(FLERR, e.what());
    }

    dedlam[i] += aceimpl->ace->dedlam_central;    // ALCHEMY

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      dedlam[j] += aceimpl->ace->dedlam_neighbours(jj);    // ALCHEMY
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];

      fij[0] = scale[itype][itype] * aceimpl->ace->neighbours_forces(jj, 0);
      fij[1] = scale[itype][itype] * aceimpl->ace->neighbours_forces(jj, 1);
      fij[2] = scale[itype][itype] * aceimpl->ace->neighbours_forces(jj, 2);

      f[i][0] += fij[0];
      f[i][1] += fij[1];
      f[i][2] += fij[2];
      f[j][0] -= fij[0];
      f[j][1] -= fij[1];
      f[j][2] -= fij[2];

      if (vflag_either) {
        ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, fij[0], fij[1], fij[2], delx, dely,
                     delz);

        if (cvflag_atom) {
          const double fx = fij[0];
          const double fy = fij[1];
          const double fz = fij[2];

          cvatom[i][0] += 0.5 * delx * fx;
          cvatom[i][1] += 0.5 * dely * fy;
          cvatom[i][2] += 0.5 * delz * fz;
          cvatom[i][3] += 0.5 * delx * fy;
          cvatom[i][4] += 0.5 * delx * fz;
          cvatom[i][5] += 0.5 * dely * fz;
          cvatom[i][6] += 0.5 * dely * fx;
          cvatom[i][7] += 0.5 * delz * fx;
          cvatom[i][8] += 0.5 * delz * fy;

          cvatom[j][0] += 0.5 * delx * fx;
          cvatom[j][1] += 0.5 * dely * fy;
          cvatom[j][2] += 0.5 * delz * fz;
          cvatom[j][3] += 0.5 * delx * fy;
          cvatom[j][4] += 0.5 * delx * fz;
          cvatom[j][5] += 0.5 * dely * fz;
          cvatom[j][6] += 0.5 * dely * fx;
          cvatom[j][7] += 0.5 * delz * fx;
          cvatom[j][8] += 0.5 * delz * fy;
        }
      }
    }

    if (eflag_either) {
      evdwl = scale[itype][itype] * aceimpl->ace->e_atom;
      ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  // ALCHEMY: fold dE/dlambda contributions on ghost atoms back to owners
  if (newton_pair) comm->reverse_comm(this);
}

/* ----------------------------------------------------------------------
   reverse communication of dedlam (ghost -> owner accumulation)
------------------------------------------------------------------------- */

int PairGRACEFSAlch::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) buf[m++] = dedlam[i];
  return m;
}

void PairGRACEFSAlch::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) dedlam[list[i]] += buf[m++];
}

/* ----------------------------------------------------------------------
   per-atom access: dedlam[i] = dE_total/dlambda_i (eV)
   (lambda-force for dynamics is the NEGATIVE of this)
------------------------------------------------------------------------- */

void *PairGRACEFSAlch::extract_peratom(const char *str, int &ncol)
{
  if (strcmp(str, "dedlam") == 0) {
    ncol = 0;
    return (void *) dedlam;
  }
  return nullptr;
}

/* ----------------------------------------------------------------------
   PHASE 2 (swap-MC): sum of per-atom energies over the given LOCAL atom
   indices, evaluated with the CURRENT atom->dvector[index_lambda]. Mirrors
   the energy part of compute() (no forces, no tally) for one atom at a time.
   The caller (fix alchemical/switch) uses this twice — before and after a
   lambda swap — to get a local Metropolis dE. Because the FS energy is
   many-body, the caller MUST include the swapped atoms and their neighbours
   in `atomlist`. Requires a current neighbour list (mid-run is fine).
------------------------------------------------------------------------- */

double PairGRACEFSAlch::cluster_energy(const int *atomlist, int n)
{
  double **x = atom->x;
  int *type = atom->type;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // point the evaluator at the live lambda array (locals + ghosts)
  aceimpl->ace->lambda = atom->dvector[index_lambda];

  // size the neighbour cache to the largest jnum we will hit
  int max_jnum = 0;
  for (int k = 0; k < n; k++) {
    const int i = atomlist[k];
    if (numneigh[i] > max_jnum) max_jnum = numneigh[i];
  }
  aceimpl->ace->resize_neighbours_cache(max_jnum);
  std::vector<int> my_neigh_jlist(max_jnum > 0 ? max_jnum : 1);

  double e_sum = 0.0;
  for (int k = 0; k < n; k++) {
    const int i = atomlist[k];
    const int itype = type[i];
    const int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; ++jj) my_neigh_jlist[jj] = jlist[jj] & NEIGHMASK;
    try {
      aceimpl->ace->compute_atom(i, x, type, jnum, my_neigh_jlist.data());
    } catch (std::exception &e) {
      error->one(FLERR, e.what());
    }
    e_sum += scale[itype][itype] * aceimpl->ace->e_atom;
  }
  return e_sum;
}

/* ---------------------------------------------------------------------- */

void PairGRACEFSAlch::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(scale, n, n, "pair:scale");
  map = new int[n];
}

/* ----------------------------------------------------------------------
   global settings: pair_style grace/fs/alch <ElementA> <ElementB>
------------------------------------------------------------------------- */

void PairGRACEFSAlch::settings(int narg, char **arg)
{
  if (narg != 2)
    error->all(FLERR, "pair_style grace/fs/alch requires exactly two arguments: "
                      "the alchemical pair, e.g. 'pair_style grace/fs/alch Cu Ag'");

  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "GRACE/FS potentials require 'metal' units");

  element_A = arg[0];
  element_B = arg[1];

  if (comm->me == 0)
    utils::logmesg(lmp, "[GRACE-FS-ALCH] alchemical pair: {} <-> {}\n", element_A, element_B);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairGRACEFSAlch::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  auto potential_file_name = utils::get_potential_file_path(arg[2]);

  //load potential file
  delete aceimpl->basis_set;
  if (comm->me == 0) utils::logmesg(lmp, "[GRACE-FS-ALCH] Loading {}\n", potential_file_name);
  aceimpl->basis_set = new GRACEFSBasisSet(potential_file_name);

  map_element2type(narg - 3, arg + 3);

  delete aceimpl->ace;
  aceimpl->ace = new GRACEFSAlchEvaluator();
  aceimpl->ace->element_type_mapping.init(atom->ntypes + 1);

  const int n = atom->ntypes;
  for (int i = 1; i <= n; i++) {
    char *elemname = arg[2 + i];
    if (strcmp(elemname, "NULL") == 0) {
      aceimpl->ace->element_type_mapping(i) = -1;
      map[i] = -1;
    } else {
      SPECIES_TYPE mu = aceimpl->basis_set->get_species_index_by_name(elemname);
      if (mu != -1) {
        if (comm->me == 0)
          utils::logmesg(lmp, "[GRACE-FS-ALCH] Mapping LAMMPS atom type #{}({}) -> "
                              "ACE species type #{}\n", i, elemname, mu);
        map[i] = mu;
        aceimpl->ace->element_type_mapping(i) = mu;
      } else {
        error->all(FLERR, "[GRACE-FS-ALCH] Element {} is not supported by potential file {}",
                   elemname, potential_file_name);
      }
    }
  }

  // initialize scale factor
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) scale[i][j] = 1.0;
  }

  aceimpl->ace->set_basis(*aceimpl->basis_set);

  // ALCHEMY: resolve the alchemical pair against the loaded basis
  SPECIES_TYPE mu_a = aceimpl->basis_set->get_species_index_by_name(element_A);
  SPECIES_TYPE mu_b = aceimpl->basis_set->get_species_index_by_name(element_B);
  if (mu_a == -1 || mu_b == -1)
    error->all(FLERR, "[GRACE-FS-ALCH] alchemical pair {}/{} not covered by potential file {}",
               element_A, element_B, potential_file_name);
  try {
    aceimpl->ace->set_alchemical_pair(mu_a, mu_b);
  } catch (std::exception &e) {
    error->all(FLERR, e.what());
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairGRACEFSAlch::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style grace/fs/alch requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style grace/fs/alch requires newton pair on");

  // ALCHEMY: locate the per-atom lambda created by fix property/atom
  int flag, cols, ghost;
  index_lambda = atom->find_custom_ghost("lambda", flag, cols, ghost);
  if (index_lambda < 0 || flag != 1 || cols != 0)
    error->all(FLERR, "pair grace/fs/alch requires a per-atom scalar 'lambda': "
                      "fix <id> all property/atom d_lambda ghost yes");
  if (!ghost)
    error->all(FLERR, "fix property/atom d_lambda must use 'ghost yes' "
                      "(neighbour lambdas are read during evaluation)");

  // allocate dedlam now so consumers (fix lambda/dynamics) can resolve
  // extract_peratom("dedlam") at their init time, before the first compute
  if (atom->nmax > nmax_dedlam) {
    memory->destroy(dedlam);
    nmax_dedlam = atom->nmax;
    memory->create(dedlam, nmax_dedlam, "pair:dedlam");
    memset(dedlam, 0, nmax_dedlam * sizeof(double));
  }

  // request a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairGRACEFSAlch::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  scale[j][i] = scale[i][j];
  return aceimpl->basis_set->radial_functions.rcut;    // the same for all
}
