/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_lambda_dynamics.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixLambdaDynamics::FixLambdaDynamics(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), lam_state(nullptr), index_lambda(-1), random(nullptr)
{
  if (narg < 4) utils::missing_cmd_args(FLERR, "fix lambda/dynamics", error);

  m_lambda = utils::numeric(FLERR, arg[3], false, lmp);
  if (m_lambda <= 0.0) error->all(FLERR, "fix lambda/dynamics mass must be > 0");

  k_bias = 0.0;
  thermostat_flag = false;
  t_target = t_damp = 0.0;
  seed = 0;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "bias") == 0) {
      k_bias = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      thermostat_flag = true;
      t_target = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      t_damp = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      seed = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      if (t_damp <= 0.0 || seed <= 0)
        error->all(FLERR, "fix lambda/dynamics temp: tau and seed must be > 0");
      iarg += 4;
    } else
      error->all(FLERR, "Unknown fix lambda/dynamics keyword: {}", arg[iarg]);
  }

  if (thermostat_flag) random = new RanMars(lmp, seed + comm->me);

  // outputs: [1] KE_lambda (eV)  [2] E_bias (eV)  [3] T_lambda (K)
  vector_flag = 1;
  size_vector = 3;
  extvector = 1;
  global_freq = 1;

  // per-atom (v_lambda, f_lambda): migrate with atoms + write to restarts
  restart_peratom = 1;
  grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);
  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) lam_state[i][0] = lam_state[i][1] = 0.0;

  comm_forward = 1;    // lambda -> ghosts every step
}

/* ---------------------------------------------------------------------- */

FixLambdaDynamics::~FixLambdaDynamics()
{
  if (copymode) return;
  atom->delete_callback(id, Atom::GROW);
  atom->delete_callback(id, Atom::RESTART);
  memory->destroy(lam_state);
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixLambdaDynamics::setmask()
{
  return INITIAL_INTEGRATE | POST_FORCE | FINAL_INTEGRATE;
}

/* ---------------------------------------------------------------------- */

void FixLambdaDynamics::init()
{
  int flag, cols, ghost;
  index_lambda = atom->find_custom_ghost("lambda", flag, cols, ghost);
  if (index_lambda < 0 || flag != 1 || cols != 0 || !ghost)
    error->all(FLERR, "fix lambda/dynamics requires 'fix property/atom d_lambda ghost yes' "
                      "defined before this fix");

  int ncol = 0;
  if (!force->pair || !force->pair->extract_peratom("dedlam", ncol))
    error->all(FLERR, "fix lambda/dynamics requires a pair style providing per-atom "
                      "'dedlam' (e.g. grace/fs/alch)");

  reset_dt();
}

/* ---------------------------------------------------------------------- */

double *FixLambdaDynamics::dedlam()
{
  int ncol = 0;
  return (double *) force->pair->extract_peratom("dedlam", ncol);
}

double *FixLambdaDynamics::lambda_vec()
{
  return atom->dvector[index_lambda];
}

/* ----------------------------------------------------------------------
   assemble the lambda-force: f = -dE/dlambda - dU_bias/dlambda + Langevin
   called after the pair style computed dedlam (and at setup)
------------------------------------------------------------------------- */

void FixLambdaDynamics::setup(int vflag)
{
  post_force(vflag);
}

void FixLambdaDynamics::post_force(int /*vflag*/)
{
  double *lambda = lambda_vec();
  double *dEdl = dedlam();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  const double boltz = force->boltz;
  const double dt = update->dt;
  const double gamma = thermostat_flag ? m_lambda / t_damp : 0.0;
  const double sigma =
      thermostat_flag ? sqrt(2.0 * boltz * t_target * m_lambda / (t_damp * dt)) : 0.0;

  double ebias_local = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    const double lam = lambda[i];

    // endpoint-localizing bias U = k_b * lam^2 (1-lam)^2
    const double fbias = -2.0 * k_bias * lam * (1.0 - lam) * (1.0 - 2.0 * lam);
    ebias_local += k_bias * lam * lam * (1.0 - lam) * (1.0 - lam);

    double f = -dEdl[i] + fbias;
    if (thermostat_flag) f += -gamma * lam_state[i][0] + sigma * random->gaussian();
    lam_state[i][1] = f;
  }
  vector_local[1] = ebias_local;
}

/* ----------------------------------------------------------------------
   velocity-Verlet on (lambda, v): half-kick + drift, then ghost update
------------------------------------------------------------------------- */

void FixLambdaDynamics::initial_integrate(int /*vflag*/)
{
  double *lambda = lambda_vec();
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    lam_state[i][0] += dtf * lam_state[i][1] / m_lambda;
    lambda[i] += dtv * lam_state[i][0];
  }

  comm->forward_comm(this);    // refresh ghost lambdas
}

void FixLambdaDynamics::final_integrate()
{
  const int *mask = atom->mask;
  const int nlocal = atom->nlocal;

  double ke_local = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    lam_state[i][0] += dtf * lam_state[i][1] / m_lambda;
    ke_local += 0.5 * m_lambda * lam_state[i][0] * lam_state[i][0];
  }
  vector_local[0] = ke_local;
}

/* ---------------------------------------------------------------------- */

double FixLambdaDynamics::compute_vector(int n)
{
  // [0] KE_lambda  [1] E_bias  [2] T_lambda
  double local[2] = {vector_local[0], vector_local[1]};
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, world);

  if (n == 0) return global[0];
  if (n == 1) return global[1];

  bigint ngroup = group->count(igroup);
  if (ngroup == 0) return 0.0;
  return 2.0 * global[0] / ((double) ngroup * force->boltz);
}

/* ---------------------------------------------------------------------- */

void FixLambdaDynamics::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt;
}

/* ----------------------------------------------------------------------
   per-atom storage management (v_lambda, f_lambda)
------------------------------------------------------------------------- */

double FixLambdaDynamics::memory_usage()
{
  return (double) atom->nmax * 2 * sizeof(double);
}

void FixLambdaDynamics::grow_arrays(int nmax)
{
  memory->grow(lam_state, nmax, 2, "lambda/dynamics:lam_state");
}

void FixLambdaDynamics::copy_arrays(int i, int j, int /*delflag*/)
{
  lam_state[j][0] = lam_state[i][0];
  lam_state[j][1] = lam_state[i][1];
}

int FixLambdaDynamics::pack_exchange(int i, double *buf)
{
  buf[0] = lam_state[i][0];
  buf[1] = lam_state[i][1];
  return 2;
}

int FixLambdaDynamics::unpack_exchange(int nlocal, double *buf)
{
  lam_state[nlocal][0] = buf[0];
  lam_state[nlocal][1] = buf[1];
  return 2;
}

int FixLambdaDynamics::pack_restart(int i, double *buf)
{
  // buf[0] = total size of this atom's restart chunk, incl. the size slot
  buf[0] = 3;
  buf[1] = lam_state[i][0];
  buf[2] = lam_state[i][1];
  return 3;
}

void FixLambdaDynamics::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;
  // skip to the nth fix's chunk for this atom
  int m = 0;
  for (int k = 0; k < nth; k++) m += (int) extra[nlocal][m];
  m++;
  lam_state[nlocal][0] = extra[nlocal][m++];
  lam_state[nlocal][1] = extra[nlocal][m++];
}

int FixLambdaDynamics::size_restart(int /*nlocal*/)
{
  return 3;
}

int FixLambdaDynamics::maxsize_restart()
{
  return 3;
}

/* ----------------------------------------------------------------------
   forward comm: owned lambda -> ghost copies, every step
------------------------------------------------------------------------- */

int FixLambdaDynamics::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                         int * /*pbc*/)
{
  double *lambda = lambda_vec();
  for (int i = 0; i < n; i++) buf[i] = lambda[list[i]];
  return n;
}

void FixLambdaDynamics::unpack_forward_comm(int n, int first, double *buf)
{
  double *lambda = lambda_vec();
  const int last = first + n;
  for (int i = first; i < last; i++) lambda[i] = buf[i - first];
}
