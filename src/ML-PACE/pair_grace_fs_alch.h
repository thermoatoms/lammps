/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

//
// Alchemical (continuous-species) GRACE/FS pair style.
//
//   pair_style grace/fs/alch <ElementA> <ElementB>
//   pair_coeff * * FS_model.yaml <El1> <El2> ...
//
// Requires a per-atom lambda from
//   fix <id> all property/atom d_lambda ghost yes
// Atoms whose element is ElementA or ElementB are a (1-lambda):(lambda)
// mix of A:B; all other atoms are evaluated discretely. lambda = 0 is
// exactly pure A, lambda = 1 exactly pure B (the atom's own type only
// carries the mass). See grace_fs_alch_evaluator.h for the math.
//

#ifdef PAIR_CLASS
// clang-format off
PairStyle(grace/fs/alch,PairGRACEFSAlch);
// clang-format on
#else

#ifndef LMP_PAIR_GRACEFS_ALCH_H
#define LMP_PAIR_GRACEFS_ALCH_H

#include "pair.h"

namespace LAMMPS_NS {

class PairGRACEFSAlch : public Pair {
 public:
  PairGRACEFSAlch(class LAMMPS *);

  ~PairGRACEFSAlch() override;

  void compute(int, int) override;

  void settings(int, char **) override;

  void coeff(int, char **) override;

  void init_style() override;

  double init_one(int, int) override;

  int pack_reverse_comm(int, int, double *) override;

  void unpack_reverse_comm(int, int *, double *) override;

  void *extract_peratom(const char *, int &) override;

 protected:
  struct ACEAlchImpl *aceimpl;

  virtual void allocate();

  std::string element_A, element_B;    // the alchemical pair, from pair_style
  int index_lambda = -1;               // index of d_lambda in atom->dvector

  // per-atom dE_total/dlambda_i (eV), rebuilt every compute(); reverse-
  // communicated so ghost contributions fold back onto owners. Readable
  // via extract_peratom("dedlam") — see the lambda-force contract.
  double *dedlam = nullptr;
  int nmax_dedlam = 0;

  double **scale;
};
}    // namespace LAMMPS_NS

#endif
#endif
