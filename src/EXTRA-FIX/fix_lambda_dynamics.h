/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// fix lambda/dynamics — velocity-Verlet integration of per-atom alchemical
// variables (lambda_i, v_i) with a global mass m_lambda, an MSlD-style
// endpoint-localizing bias U_b = k_b * lambda^2 (1-lambda)^2, and an
// optional Langevin thermostat on the lambda subsystem.
//
//   fix <id> <group> lambda/dynamics <m_lambda> [bias <k_b>]
//                                    [temp <T> <tau> <seed>]
//
// lambda lives in fix property/atom d_lambda (ghost yes, defined BEFORE
// this fix); dE/dlambda comes from the pair style via
// extract_peratom("dedlam") — see notes/lambda-force-contract.md.
// Design rationale and validation gates: notes/m4-bias-design.md.
//

#ifdef FIX_CLASS
// clang-format off
FixStyle(lambda/dynamics,FixLambdaDynamics);
// clang-format on
#else

#ifndef LMP_FIX_LAMBDA_DYNAMICS_H
#define LMP_FIX_LAMBDA_DYNAMICS_H

#include "fix.h"

namespace LAMMPS_NS {

class FixLambdaDynamics : public Fix {
 public:
  FixLambdaDynamics(class LAMMPS *, int, char **);
  ~FixLambdaDynamics() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void post_force(int) override;
  void final_integrate() override;
  double compute_vector(int) override;
  void reset_dt() override;

  // fix-owned per-atom storage (v_lambda, f_lambda): grow/migrate/restart
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_restart(int, double *) override;
  void unpack_restart(int, int) override;
  int size_restart(int) override;
  int maxsize_restart() override;

  // per-step ghost update of lambda (fix property/atom only refreshes
  // ghosts at reneighboring; dynamics changes lambda every step)
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  double m_lambda;     // mass of the lambda DOF, eV*ps^2
  double k_bias;       // endpoint bias strength, eV
  double t_target;     // Langevin temperature, K (thermostat_flag only)
  double t_damp;       // Langevin damping time, ps
  int seed;
  bool thermostat_flag;

  double **lam_state;    // per atom: [0] = v_lambda, [1] = f_lambda

  int index_lambda;          // d_lambda slot in atom->dvector
  class RanMars *random;
  double dtv, dtf;

  double *dedlam();          // fresh pointer from the pair style
  double *lambda_vec();      // fresh pointer to d_lambda

  // accumulated for compute_vector: [KE_lambda, E_bias, T_lambda]
  double vector_local[2];
};

}    // namespace LAMMPS_NS

#endif
#endif
