/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_alchemical_switch.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixAlchemicalSwitch::FixAlchemicalSwitch(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), order_file(nullptr), out_file(nullptr), fp(nullptr),
    switch_order(nullptr), random(nullptr), index_lambda(-1)
{
  if (narg < 11) error->all(FLERR, "Illegal fix alchemical/switch command");

  // arg[3] is the pair-style name (informational; we resolve via force->pair)
  // accept and ignore so the syntax documents which pair style is required.

  // defaults
  avg_flag = 0;
  order_random = 1;
  seed = 12345;
  gsize = 1;            // atoms switched together per group (1 = sequential)
  nswap = 0;            // swap-MC off by default
  swap_every = 0;
  swap_temp = -1.0;     // <=0 -> use simulation temperature
  nswap_attempt = nswap_accept = 0;

  int iarg = 4;
  type_A = type_B = -1;
  xtarget = -1.0;
  nsub = nrelax = -1;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "pair") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "fix alchemical/switch: 'pair' needs two types");
      type_A = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      type_B = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg], "xtarget") == 0) {
      xtarget = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "nsub") == 0) {
      nsub = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "nrelax") == 0) {
      nrelax = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "group") == 0) {
      gsize = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "seed") == 0) {
      seed = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "order") == 0) {
      if (strcmp(arg[iarg + 1], "random") == 0) {
        order_random = 1;
        iarg += 2;
      } else if (strcmp(arg[iarg + 1], "file") == 0) {
        order_random = 0;
        delete[] order_file;
        order_file = utils::strdup(arg[iarg + 2]);
        iarg += 3;
      } else
        error->all(FLERR, "fix alchemical/switch: order must be 'random' or 'file <name>'");
    } else if (strcmp(arg[iarg], "avg") == 0) {
      avg_flag = (strcmp(arg[iarg + 1], "yes") == 0) ? 1 : 0;
      iarg += 2;
    } else if (strcmp(arg[iarg], "out") == 0) {
      delete[] out_file;
      out_file = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "nswap") == 0) {
      nswap = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "swapevery") == 0) {
      swap_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "swaptemp") == 0) {
      swap_temp = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix alchemical/switch keyword: {}", arg[iarg]);
  }

  if (type_A < 0 || type_B < 0) error->all(FLERR, "fix alchemical/switch: 'pair <A> <B>' required");
  if (xtarget < 0.0) error->all(FLERR, "fix alchemical/switch: 'xtarget <x>' required");
  if (nsub <= 0) error->all(FLERR, "fix alchemical/switch: 'nsub > 0' required");
  if (nrelax <= 0) error->all(FLERR, "fix alchemical/switch: 'nrelax > 0' required");
  if (gsize <= 0) error->all(FLERR, "fix alchemical/switch: 'group >= 1' required");
  if (nswap > 0 && swap_every <= 0)
    error->all(FLERR, "fix alchemical/switch: nswap>0 requires swapevery>0");
  if (nswap > 0 && swap_temp <= 0.0)
    error->all(FLERR, "fix alchemical/switch: nswap>0 requires swaptemp>0 (K)");

  // rank-IDENTICAL stream: used by attempt_swaps so every rank makes the same
  // choices (the order shuffle in build_order uses its own local RNG).
  random = new RanMars(lmp, seed);

  // outputs: [nswitched, x, F(x), nswap_attempt, nswap_accept, accept_ratio]
  vector_flag = 1;
  size_vector = 6;
  global_freq = 1;   // values are valid every step (counters + running F)
  extvector = 0;

  comm_forward = 1;    // lambda -> ghosts every step

  // runtime state
  switch_order = nullptr;
  nswitch = 0;
  cur_atom = 0;
  cur_sub = 0;
  prev_lambda = prev_dedlam = 0.0;
  W_atom = W_cum = 0.0;
  nswitched = 0;
  dedlam_accum = 0.0;
  dedlam_nsamp = 0;
}

/* ---------------------------------------------------------------------- */

FixAlchemicalSwitch::~FixAlchemicalSwitch()
{
  delete random;
  delete[] order_file;
  delete[] out_file;
  memory->destroy(switch_order);
  if (fp && comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixAlchemicalSwitch::setmask()
{
  return END_OF_STEP;
}

/* ---------------------------------------------------------------------- */

void FixAlchemicalSwitch::init()
{
  int flag, cols, ghost;
  index_lambda = atom->find_custom_ghost("lambda", flag, cols, ghost);
  if (index_lambda < 0 || flag != 1 || cols != 0 || !ghost)
    error->all(FLERR, "fix alchemical/switch requires 'fix property/atom d_lambda ghost yes' "
                      "defined before this fix");

  int ncol = 0;
  if (!force->pair || !force->pair->extract_peratom("dedlam", ncol))
    error->all(FLERR, "fix alchemical/switch requires a pair style providing per-atom "
                      "'dedlam' (e.g. grace/fs/alch)");

  if (atom->tag_enable == 0)
    error->all(FLERR, "fix alchemical/switch requires atom IDs");
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "fix alchemical/switch requires an atom map (atom_modify map yes)");
}

/* ---------------------------------------------------------------------- */

double *FixAlchemicalSwitch::dedlam()
{
  int ncol = 0;
  return (double *) force->pair->extract_peratom("dedlam", ncol);
}

double *FixAlchemicalSwitch::lambda_vec()
{
  return atom->dvector[index_lambda];
}

/* ----------------------------------------------------------------------
   build the switch schedule: which atoms (global IDs) and the direction.
   Switch atoms of species A toward xtarget. nswitch = round(|x0-xt|*N).
   Direction: if xtarget < x0(A) we convert A->B (set their lambda 0->1);
   if xtarget > x0(A) we convert B->A (set their lambda 1->0). Atoms start
   at lambda matching their type (A: 0, B: 1) — set in setup().
------------------------------------------------------------------------- */

void FixAlchemicalSwitch::build_order()
{
  N = atom->natoms;

  // count species A and B globally
  const int *type = atom->type;
  const int nlocal = atom->nlocal;
  bigint nA_local = 0, nB_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (type[i] == type_A) nA_local++;
    else if (type[i] == type_B) nB_local++;
  }
  bigint nA = 0, nB = 0;
  MPI_Allreduce(&nA_local, &nA, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Allreduce(&nB_local, &nB, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  const double x0 = (double) nA / (double) N;
  const double dx = xtarget - x0;
  nswitch = (int) std::lround(std::fabs(dx) * (double) N);

  // direction of the species we convert FROM:
  //   xtarget < x0(A): convert A->B  -> switch A atoms, lambda 0->1  (dir=+1)
  //   xtarget > x0(A): convert B->A  -> switch B atoms, lambda 1->0  (dir=-1)
  dir = (dx <= 0.0) ? +1 : -1;
  const int from_type = (dir == +1) ? type_A : type_B;

  if (nswitch == 0) {
    if (comm->me == 0)
      error->warning(FLERR, "fix alchemical/switch: xtarget equals current composition; "
                            "nothing to switch");
    return;
  }
  if (dir == +1 && nswitch > nA)
    error->all(FLERR, "fix alchemical/switch: not enough A atoms to reach xtarget");
  if (dir == -1 && nswitch > nB)
    error->all(FLERR, "fix alchemical/switch: not enough B atoms to reach xtarget");

  // gather all global IDs of from_type onto every rank (small: O(N) ints)
  tagint *local_ids;
  memory->create(local_ids, (nlocal > 0 ? nlocal : 1), "alch/switch:local_ids");
  int ncand_local = 0;
  const tagint *tag = atom->tag;
  for (int i = 0; i < nlocal; i++)
    if (type[i] == from_type) local_ids[ncand_local++] = tag[i];

  int nproc = comm->nprocs;
  int *counts = new int[nproc];
  int *displs = new int[nproc];
  MPI_Allgather(&ncand_local, 1, MPI_INT, counts, 1, MPI_INT, world);
  int ntot = 0;
  for (int p = 0; p < nproc; p++) {
    displs[p] = ntot;
    ntot += counts[p];
  }
  tagint *all_ids;
  memory->create(all_ids, (ntot > 0 ? ntot : 1), "alch/switch:all_ids");
  MPI_Allgatherv(local_ids, ncand_local, MPI_LMP_TAGINT, all_ids, counts, displs, MPI_LMP_TAGINT,
                 world);

  // deterministic order independent of rank count: sort the candidate IDs,
  // then shuffle with a serial RNG seeded by `seed` (identical on all ranks).
  std::sort(all_ids, all_ids + ntot);

  if (order_random) {
    // Fisher-Yates with a per-rank-identical stream (seed only, no +me here)
    RanMars rng(lmp, seed);
    for (int i = ntot - 1; i > 0; i--) {
      int j = (int) (rng.uniform() * (i + 1));
      if (j > i) j = i;
      tagint tmp = all_ids[i];
      all_ids[i] = all_ids[j];
      all_ids[j] = tmp;
    }
  }
  // (order from file: handled below, overrides the shuffle)

  memory->create(switch_order, nswitch, "alch/switch:order");
  if (!order_random && order_file) {
    // read first nswitch IDs from file on rank 0, broadcast
    if (comm->me == 0) {
      FILE *f = fopen(order_file, "r");
      if (!f) error->one(FLERR, "fix alchemical/switch: cannot open order file {}", order_file);
      for (int k = 0; k < nswitch; k++) {
        long id;
        if (fscanf(f, "%ld", &id) != 1)
          error->one(FLERR, "fix alchemical/switch: order file too short");
        switch_order[k] = (tagint) id;
      }
      fclose(f);
    }
    MPI_Bcast(switch_order, nswitch, MPI_LMP_TAGINT, 0, world);
  } else {
    for (int k = 0; k < nswitch; k++) switch_order[k] = all_ids[k];
  }

  memory->destroy(local_ids);
  memory->destroy(all_ids);
  delete[] counts;
  delete[] displs;
}

/* ----------------------------------------------------------------------
   set d_lambda on the owner of global atom `id`; no-op on ranks that don't
   own it. Caller must follow with comm->forward_comm(this) to refresh ghosts.
------------------------------------------------------------------------- */

void FixAlchemicalSwitch::set_lambda(tagint id, double val)
{
  const int ilocal = atom->map(id);
  if (ilocal >= 0 && ilocal < atom->nlocal) lambda_vec()[ilocal] = val;
}

/* ----------------------------------------------------------------------
   number of atoms in the current group: gsize, except the last group may be
   partial when nswitch is not a multiple of gsize.
------------------------------------------------------------------------- */

int FixAlchemicalSwitch::group_len() const
{
  int rem = nswitch - cur_atom;
  return (rem < gsize) ? rem : gsize;
}

/* ----------------------------------------------------------------------
   set d_lambda = val on every atom of the active group (shared lambda), then
   refresh ghosts.
------------------------------------------------------------------------- */

void FixAlchemicalSwitch::set_group_lambda(double val)
{
  const int g = group_len();
  for (int m = 0; m < g; m++) set_lambda(switch_order[cur_atom + m], val);
  comm->forward_comm(this);
}

/* ----------------------------------------------------------------------
   MPI-reduce the SUM of dedlam over the active group: each owner contributes
   its members' values, all reduced and broadcast. For gsize=1 this is the
   single active atom (unchanged behaviour).
------------------------------------------------------------------------- */

double FixAlchemicalSwitch::active_dedlam()
{
  const int g = group_len();
  double *dl = dedlam();
  const int nlocal = atom->nlocal;
  double local = 0.0;
  for (int m = 0; m < g; m++) {
    const int ilocal = atom->map(switch_order[cur_atom + m]);
    if (ilocal >= 0 && ilocal < nlocal) local += dl[ilocal];
  }
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  return global;
}

/* ----------------------------------------------------------------------
   total pair energy (eV), MPI-summed. Bulletproof (atom/swap style): drive a
   full pair compute with eflag and reduce eng_vdwl. Slow per call (full force
   eval) but guaranteed MPI-correct for the many-body FS energy. A future fast
   path can use pair->cluster_energy over the affected local cluster instead.
------------------------------------------------------------------------- */

double FixAlchemicalSwitch::energy_full()
{
  comm->forward_comm(this);           // make sure ghosts see current lambda
  if (modify->n_pre_force) modify->pre_force(0);
  if (force->pair) force->pair->compute(1, 0);   // eflag=1, vflag=0
  if (modify->n_post_force_any) modify->post_force(0);
  double e_local = force->pair ? force->pair->eng_vdwl : 0.0;
  double e_global = 0.0;
  MPI_Allreduce(&e_local, &e_global, 1, MPI_DOUBLE, MPI_SUM, world);
  return e_global;
}

/* ----------------------------------------------------------------------
   one swap cycle: nswap Metropolis attempts. Each attempt exchanges lambda
   between one lambda=1 atom and one lambda=0 atom (FIXED composition) and
   accepts on exp(-dE/kT). All random choices are made identically on every
   rank (rank-independent stream) so the schedule is MPI-consistent; the
   energy is the MPI-summed full pair energy.

   The pools (lambda~=1 ids, lambda~=0 ids) are rebuilt each cycle by gathering
   the alchemical-pair atoms' current lambda globally.
------------------------------------------------------------------------- */

void FixAlchemicalSwitch::attempt_swaps()
{
  const int nlocal = atom->nlocal;
  const int *type = atom->type;
  const tagint *tag = atom->tag;
  double *lambda = lambda_vec();

  // gather (id, lambda) for all alchemical-pair atoms onto every rank
  tagint *lid;  double *llam;
  memory->create(lid,  (nlocal > 0 ? nlocal : 1), "alch/swap:lid");
  memory->create(llam, (nlocal > 0 ? nlocal : 1), "alch/swap:llam");
  int nloc = 0;
  for (int i = 0; i < nlocal; i++) {
    if (type[i] == type_A || type[i] == type_B) {
      lid[nloc] = tag[i]; llam[nloc] = lambda[i]; nloc++;
    }
  }
  const int nproc = comm->nprocs;
  int *cnt = new int[nproc]; int *dsp = new int[nproc];
  MPI_Allgather(&nloc, 1, MPI_INT, cnt, 1, MPI_INT, world);
  int ntot = 0;
  for (int p = 0; p < nproc; p++) { dsp[p] = ntot; ntot += cnt[p]; }
  tagint *gid;  double *glam;
  memory->create(gid,  (ntot > 0 ? ntot : 1), "alch/swap:gid");
  memory->create(glam, (ntot > 0 ? ntot : 1), "alch/swap:glam");
  MPI_Allgatherv(lid,  nloc, MPI_LMP_TAGINT, gid,  cnt, dsp, MPI_LMP_TAGINT, world);
  MPI_Allgatherv(llam, nloc, MPI_DOUBLE,     glam, cnt, dsp, MPI_DOUBLE,     world);

  // build pools by current lambda (endpoint atoms only; the single transiting
  // atom of the switch schedule has 0<lambda<1 and is excluded from both)
  int n1 = 0, n0 = 0;
  for (int k = 0; k < ntot; k++) {
    if (glam[k] > 0.999) n1++;
    else if (glam[k] < 0.001) n0++;
  }
  tagint *pool1 = new tagint[n1 > 0 ? n1 : 1];
  tagint *pool0 = new tagint[n0 > 0 ? n0 : 1];
  int i1 = 0, i0 = 0;
  for (int k = 0; k < ntot; k++) {
    if (glam[k] > 0.999) pool1[i1++] = gid[k];
    else if (glam[k] < 0.001) pool0[i0++] = gid[k];
  }

  const double kT = force->boltz * swap_temp;

  if (n1 > 0 && n0 > 0) {
    for (int s = 0; s < nswap; s++) {
      nswap_attempt++;
      // identical choices on all ranks
      const int a = (int) (random->uniform() * n1);
      const int b = (int) (random->uniform() * n0);
      const tagint id1 = pool1[a < n1 ? a : n1 - 1];
      const tagint id0 = pool0[b < n0 ? b : n0 - 1];

      const double e_before = energy_full();
      // swap: id1 -> 0, id0 -> 1
      set_lambda(id1, 0.0);
      set_lambda(id0, 1.0);
      const double e_after = energy_full();
      const double dE = e_after - e_before;

      // accept/reject with a rank-identical random number
      const double r = random->uniform();
      bool accept = (dE <= 0.0) || (r < std::exp(-dE / kT));
      if (accept) {
        nswap_accept++;
        // keep the swap; update the pools in place so subsequent attempts see it
        pool1[a < n1 ? a : n1 - 1] = id0;   // id0 is now lambda=1
        pool0[b < n0 ? b : n0 - 1] = id1;   // id1 is now lambda=0
      } else {
        set_lambda(id1, 1.0);   // revert
        set_lambda(id0, 0.0);
        comm->forward_comm(this);
      }
    }
  }

  delete[] pool1; delete[] pool0;
  delete[] cnt; delete[] dsp;
  memory->destroy(lid); memory->destroy(llam);
  memory->destroy(gid); memory->destroy(glam);
}

/* ----------------------------------------------------------------------
   setup: build schedule, initialise all lambdas to their type endpoints,
   place the first switching atom at lambda=0 (its starting endpoint), and
   seed the trapezoid with the lambda=0 dedlam.
------------------------------------------------------------------------- */

void FixAlchemicalSwitch::setup(int /*vflag*/)
{
  build_order();

  // initialise lambda for ALL atoms to their type endpoint:
  //   type_A -> lambda 0, type_B -> lambda 1, others left as-is.
  double *lambda = lambda_vec();
  const int *type = atom->type;
  for (int i = 0; i < atom->nlocal; i++) {
    if (type[i] == type_A) lambda[i] = 0.0;
    else if (type[i] == type_B) lambda[i] = 1.0;
  }
  comm->forward_comm(this);

  if (nswitch == 0) return;

  t_start = update->ntimestep;
  cur_atom = 0;
  cur_sub = 0;
  nswitched = 0;
  W_atom = W_cum = 0.0;

  // open log on rank 0
  if (out_file && comm->me == 0) {
    fp = fopen(out_file, "w");
    if (!fp) error->one(FLERR, "fix alchemical/switch: cannot open out file {}", out_file);
    fprintf(fp, "# nswitched group_size lambda dedlam Wcum F   "
                "(one row per completed group; x=nswitched/N)\n");
  }

  // the first group begins at its starting endpoint:
  //   dir=+1 (A->B): start lambda=0; dir=-1 (B->A): start lambda=1.
  const double lam0 = (dir == +1) ? 0.0 : 1.0;
  set_group_lambda(lam0);

  // need the pair style's dedlam at this state; the integrator will run a
  // force eval for step 0. Seed the trapezoid baseline now (sum over the group).
  prev_lambda = lam0;
  prev_dedlam = active_dedlam();
  W_atom = 0.0;
  dedlam_accum = 0.0;
  dedlam_nsamp = 0;
}

/* ----------------------------------------------------------------------
   end_of_step: every nrelax steps we have finished a relax window for the
   current (atom, substep). Read dedlam, trapezoid-integrate from the previous
   point, advance lambda to the next substep (or move to the next atom).
------------------------------------------------------------------------- */

void FixAlchemicalSwitch::end_of_step()
{
  // PHASE 2: swap-MC cycle (independent of switch progress; runs during AND
  // after switching so the configuration can equilibrate at fixed composition)
  if (nswap > 0 && swap_every > 0 && (update->ntimestep % swap_every == 0))
    attempt_swaps();

  if (nswitch == 0) return;
  if (nswitched >= nswitch) return;    // schedule complete

  // accumulate for block-averaging within the window
  if (avg_flag) {
    dedlam_accum += active_dedlam();
    dedlam_nsamp++;
  }

  // only act at the end of each relax window
  const bigint elapsed = update->ntimestep - t_start;
  if (elapsed % nrelax != 0) return;

  // <dE/dlambda> at the current (just-finished) lambda
  double cur_dedlam;
  if (avg_flag && dedlam_nsamp > 0) {
    cur_dedlam = dedlam_accum / dedlam_nsamp;
  } else {
    cur_dedlam = active_dedlam();
  }
  dedlam_accum = 0.0;
  dedlam_nsamp = 0;

  const int g = group_len();

  // current (shared) lambda of the active group (endpoint of the window we ran)
  const double lam_now = (dir == +1) ? (double) cur_sub / nsub
                                     : 1.0 - (double) cur_sub / nsub;

  // trapezoid increment of the SUMMED group dedlam over the common lambda
  if (cur_sub > 0) {
    const double dl = lam_now - prev_lambda;
    W_atom += 0.5 * (cur_dedlam + prev_dedlam) * dl;
  }
  prev_lambda = lam_now;
  prev_dedlam = cur_dedlam;

  // advance
  if (cur_sub == nsub) {
    // group fully switched: commit its work and step the composition by g atoms.
    // F(x) is logged ONLY here -> x lands on a whole-atom grid (g, 2g, ... /N).
    W_cum += W_atom;
    nswitched += g;
    cur_atom += g;
    cur_sub = 0;
    W_atom = 0.0;
    if (fp && comm->me == 0)
      fprintf(fp, "%d %d %.4f %.10g %.10g %.10g\n", nswitched, g, 1.0,
              cur_dedlam, W_cum, W_cum / (double) N);
    if (nswitched >= nswitch) return;
    // place next group at its starting endpoint
    const double lam0 = (dir == +1) ? 0.0 : 1.0;
    set_group_lambda(lam0);
    prev_lambda = lam0;
    prev_dedlam = active_dedlam();
  } else {
    // advance the whole group's shared lambda by one increment
    cur_sub++;
    const double lam_next = (dir == +1) ? (double) cur_sub / nsub
                                        : 1.0 - (double) cur_sub / nsub;
    set_group_lambda(lam_next);
  }
}

/* ---------------------------------------------------------------------- */

double FixAlchemicalSwitch::compute_vector(int n)
{
  if (n == 0) return (double) nswitched;
  if (n == 1) return (double) nswitched / (double) N;
  if (n == 2) return (W_cum) / (double) N;    // F(x) over fully-switched atoms
  if (n == 3) return (double) nswap_attempt;
  if (n == 4) return (double) nswap_accept;
  return nswap_attempt > 0 ? (double) nswap_accept / nswap_attempt : 0.0;  // accept ratio
}

/* ----------------------------------------------------------------------
   forward comm: owned lambda -> ghost copies (we change lambda mid-run)
------------------------------------------------------------------------- */

int FixAlchemicalSwitch::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                           int * /*pbc*/)
{
  double *lambda = lambda_vec();
  for (int i = 0; i < n; i++) buf[i] = lambda[list[i]];
  return n;
}

void FixAlchemicalSwitch::unpack_forward_comm(int n, int first, double *buf)
{
  double *lambda = lambda_vec();
  const int last = first + n;
  for (int i = first; i < last; i++) lambda[i] = buf[i - first];
}
