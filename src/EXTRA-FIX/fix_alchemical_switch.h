/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// fix alchemical/switch — sequential discrete alchemical switching (M5c).
//
// Switch atoms one-at-a-time from species A to species B by ramping each
// atom's continuous lambda 0 -> 1 over nsub increments, with nrelax MD
// steps of equilibration between increments. Integrate the switch work
// W_k = integral_0^1 <dE/dlambda_k> dlambda for each switched atom k online
// (trapezoid), accumulate the running free energy F(x = k/N) = (1/N) sum W.
// lambda is ALWAYS 0 or 1 except the single transiting atom -> the system
// is a real (discrete) alloy at every step, never a mean-field smear.
//
//   fix <id> <group> alchemical/switch <pairstyle> \
//       pair <typeA> <typeB> xtarget <x> \
//       nsub <n> nrelax <m> seed <s> \
//       [order random|file <fname>] [avg yes|no] [out <fname>]
//
//   pair <A> <B> xtarget <x> : drive the global mole fraction of species A
//       toward target x by converting A<->B. nswitch = round(|x - x0| * N),
//       direction inferred from sign(x - x0). A is the first named type.
//   nsub   : lambda increments per atom (lambda = 1/nsub .. 1).
//   nrelax : MD steps run between lambda increments (the convergence knob;
//            under-relaxed -> dissipation, gap grows with x).
//   order  : random (seeded shuffle, default) or a file of atom IDs.
//   avg    : no  -> dedlam read at last step of each nrelax window (default,
//                   reproduces validated runs);
//            yes -> dedlam block-averaged over the nrelax window.
//   out    : per-substep log "k atom lambda dedlam Wcum F" (rank 0 only).
//
// This is timestep-driven: issue ONE run of length nswitch*(nsub+1)*nrelax;
// the fix advances lambda every nrelax steps on end_of_step. No restart
// (phase 1). lambda lives in fix property/atom d_lambda (ghost yes, defined
// BEFORE this fix); dE/dlambda comes from the pair style via
// extract_peratom("dedlam"). See fix_lambda_dynamics for the shared idioms.
//
// Output (compute_vector): [0] nswitched, [1] x = nswitched/N, [2] F(x).
//

#ifdef FIX_CLASS
// clang-format off
FixStyle(alchemical/switch,FixAlchemicalSwitch);
// clang-format on
#else

#ifndef LMP_FIX_ALCHEMICAL_SWITCH_H
#define LMP_FIX_ALCHEMICAL_SWITCH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAlchemicalSwitch : public Fix {
 public:
  FixAlchemicalSwitch(class LAMMPS *, int, char **);
  ~FixAlchemicalSwitch() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;
  double compute_vector(int) override;

  // per-step ghost update of lambda (fix property/atom only refreshes ghosts
  // at reneighboring; we change lambda mid-run)
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  int type_A, type_B;        // alchemical pair: switch A -> B (or B -> A)
  double xtarget;            // target global mole fraction of species A
  int nsub;                  // lambda increments per atom
  int nrelax;                // MD steps between increments
  int seed;
  int avg_flag;              // 0 = last-step snapshot, 1 = block-average dedlam
  int order_random;          // 1 = seeded shuffle, 0 = from file
  char *order_file;          // atom-ID list if order_random == 0
  char *out_file;            // per-substep log (rank 0), or nullptr
  FILE *fp;                  // open handle for out_file (rank 0)

  // schedule
  tagint *switch_order;      // global atom IDs to switch, length nswitch
  int nswitch;               // number of atoms to switch (sets x)
  int dir;                   // +1: A->B (lambda 0->1); switching reduces x(A)

  bigint N;                  // total atom count (for x = k/N)
  bigint t_start;            // update->ntimestep at setup (schedule origin)

  // running integration state
  int cur_atom;              // index into switch_order of the atom now switching
  int cur_sub;               // current lambda increment (0..nsub) for cur_atom
  double prev_lambda;        // lambda at previous logged point (trapezoid)
  double prev_dedlam;        // <dE/dlambda> at previous logged point
  double W_atom;             // accumulated work for the current atom
  double W_cum;              // accumulated work over all completed atoms
  int    nswitched;          // atoms fully switched so far

  // block-average accumulators (avg_flag == 1)
  double dedlam_accum;       // sum of dedlam over the relax window
  int    dedlam_nsamp;       // samples in the window

  int index_lambda;          // d_lambda slot in atom->dvector
  class RanMars *random;

  double *dedlam();          // fresh pointer from the pair style
  double *lambda_vec();      // fresh pointer to d_lambda

  double active_dedlam();    // MPI_Allreduce'd <dE/dlambda> of the active atom
  void   set_lambda(tagint id, double val);   // set d_lambda on the owner
  void   build_order();      // construct switch_order + direction
};

}    // namespace LAMMPS_NS

#endif
#endif
