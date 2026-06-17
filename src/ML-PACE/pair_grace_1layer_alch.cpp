/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*
Copyright 2025 Yury Lysogorskiy^1,  Anton Bochkarev^1, Ralf Drautz^1

^1: Ruhr-University Bochum, Bochum, Germany
*/

#ifndef NO_GRACE_TF
// #define GRACE_CHUNK_DEBUG
// #define GRACE_PROFILE

#include "pair_grace_1layer_alch.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include "yaml-cpp/yaml.h"
#include <cstring>
#include <numeric>

#include "utils_grace.h"
#include "utils_pace.h"

// CppFlow headers
#include <cppflow/model.h>
#include <cppflow/ops.h>
#include <cppflow/tensor.h>
#include <string>
#include <tensorflow/c/c_api.h>
#include <unistd.h>

namespace LAMMPS_NS {

struct GRACE1LayerAlchImpl {
  cppflow::model *model = nullptr;
  GRACE::GracePaddingDimension atom_padding;
  GRACE::GracePaddingDimension real_atom_padding;
  GRACE::GracePaddingDimension neighbor_padding;

  std::map<std::string, cppflow::TensorInfo> compute_inputs_sig;
  std::map<std::string, cppflow::TensorInfo> compute_energy_only_inputs_sig;

  // Persistent buffers for efficient indexing and marshalling
  std::vector<int> global_to_chunk_map;    // size nall, init to -1
  std::vector<int> chunk_to_global_map;    // size chunksize + buffer

  std::vector<int32_t> atomic_mu_i_local;    // size n_real_padded
  std::vector<int32_t> ind_i;
  std::vector<int32_t> ind_j;
  std::vector<int32_t> mu_i;
  std::vector<int32_t> mu_j;
  std::vector<double> bond_vector;
  std::vector<int32_t> map_atoms_to_structure;

  // -- ALCHEMY: per-(local)node lambda + the A/B element indices it morphs --
  std::vector<double> lambda_i;     // size n_real_padded (or n_nodes_padded)
  std::vector<int32_t> mu_a_i;      // "from" element index per node
  std::vector<int32_t> mu_b_i;      // "to"   element index per node

  bool graph_recompiled = false;

  GRACE1LayerAlchImpl() = default;
  ~GRACE1LayerAlchImpl() { delete model; }
};

}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairGRACE1LayerAlch::PairGRACE1LayerAlch(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  comm_reverse = 1;          // ALCHEMY: reverse-comm dedlam ghost->owner

  impl = new GRACE1LayerAlchImpl;

  scale = nullptr;
  chunksize = 4096;

  total_timer.init();
  data_timer.init();
  model_timer.init();
  tp_timer.init();

  no_virial_fdotr_compute = 1;
  flag_compute_energy_only = 0;
}

/* ---------------------------------------------------------------------- */

PairGRACE1LayerAlch::~PairGRACE1LayerAlch()
{
  if (copymode) return;

  if (comm->me == 0 && total_real_atoms_processed > 0) {
    GRACE::log_perf_stats(lmp, "grace/1layer/chunk", total_real_atoms_processed,
                          total_compute_calls,
                          {{"Total", total_timer.as_microseconds()},
                           {"Data", data_timer.as_microseconds()},
                           {"Model", model_timer.as_microseconds()},
                           {"TP", tp_timer.as_microseconds()}});
  }

  delete impl;

  memory->destroy(dedlam);      // ALCHEMY

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairGRACE1LayerAlch::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(scale, n, n, "pair:scale");
  map = new int[n];
}

/* ---------------------------------------------------------------------- */

void PairGRACE1LayerAlch::settings(int narg, char **arg)
{
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "GRACE potentials require 'metal' units");

  if (comm->me == 0) utils::logmesg(lmp, "[GRACE] TF version: {}\n", TF_Version());

  // ALCHEMY: pair_style grace/1layer/alch <ElementA> <ElementB> [keywords...]
  if (narg < 2)
    error->all(FLERR, "pair_style grace/1layer/alch requires two element names "
                      "(the alchemical A B pair) before any keywords");
  element_A = arg[0];
  element_B = arg[1];

  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "chunksize") == 0) {
      chunksize = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "padding") == 0) {
      neigh_padding_fraction = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "pad_verbose") == 0) {
      pad_verbose = true;
      iarg += 1;
    } else if (strcmp(arg[iarg], "max_number_of_reduction") == 0) {
      max_number_of_reduction = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "reduce_padding") == 0) {
      reducing_neigh_padding_fraction = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "debug_no_energy_only_calc") == 0) {
      debug_no_energy_only_calc = true;
      iarg += 1;
    } else
      error->all(FLERR, "[GRACE] Unknown pair_style grace/1layer/chunk keyword: {}", arg[iarg]);
  }

  do_padding = (neigh_padding_fraction > 0);

  // Configure padding helpers
  impl->atom_padding.enabled = do_padding;
  impl->atom_padding.padding_fraction = neigh_padding_fraction;
  impl->atom_padding.reduction_threshold_fraction = reducing_neigh_padding_fraction;
  impl->atom_padding.max_reductions = max_number_of_reduction;
  impl->atom_padding.verbose = pad_verbose;

  impl->real_atom_padding = impl->atom_padding;
  impl->neighbor_padding = impl->atom_padding;

  centroidstressflag = CENTROID_AVAIL;
}

/* ---------------------------------------------------------------------- */

void PairGRACE1LayerAlch::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  map_element2type(narg - 3, arg + 3);

  auto potential_path = std::string(arg[2]);

  if (impl->model) {
    delete impl->model;
    impl->model = nullptr;
  }

  if (comm->me == 0) utils::logmesg(lmp, "[GRACE] Loading {}\n", potential_path);

  const std::vector<uint8_t> config_bytes = {0x32, 0x05, 0x82, 0x01, 0x02, 0x18, 0x00};

  impl->model = new cppflow::model(potential_path, config_bytes);
  if (comm->me == 0) std::cerr << "[GRACE] model loaded" << std::endl;

  YAML_PACE::Node metadata_yaml = YAML_PACE::LoadFile(potential_path + "/metadata.yaml");
  elements_name = metadata_yaml["chemical_symbols"].as<std::vector<std::string>>();
  nelements = (int) elements_name.size();
  for (int mu = 0; mu < nelements; mu++) { elements_to_index_map[elements_name.at(mu)] = mu; }
  cutoff = metadata_yaml["cutoff"].as<double>();

  if (metadata_yaml["cutoff_matrix"]) {
    cutoff_matrix = metadata_yaml["cutoff_matrix"].as<std::vector<std::vector<double>>>();
    is_custom_cutoffs = true;
  }

  const int ntypes = atom->ntypes;
  element_type_mapping.resize(ntypes + 1);
  for (int i = 1; i <= ntypes; i++) {
    char *elemname = arg[2 + i];
    if (strcmp(elemname, "NULL") == 0) {
      element_type_mapping[i] = -1;
    } else {
      int mu = elements_to_index_map.at(elemname);
      element_type_mapping[i] = mu;
    }
  }

  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) scale[i][j] = 1.0;
  }

  if (is_custom_cutoffs) {
    cutoff_matrix_per_lammps_type.resize(ntypes + 1, std::vector<double>(ntypes + 1));
    for (int i = 1; i <= ntypes; i++) {
      for (int j = 1; j <= ntypes; j++) {
        cutoff_matrix_per_lammps_type[i][j] =
            cutoff_matrix[element_type_mapping[i]][element_type_mapping[j]];
      }
    }
  }

  if (impl->model->has_signature("compute")) {
    compute_function_name = "compute";
  } else if (impl->model->has_signature("serving_default")) {
    compute_function_name = "serving_default";
  }
  impl->compute_inputs_sig = impl->model->signatures.at(compute_function_name).inputs;

  if (impl->model->has_signature(COMPUTE_ENERGY_ONLY_KEY)) {
    compute_energy_only_function_name = COMPUTE_ENERGY_ONLY_KEY;
    has_compute_energy_only = true;
    impl->compute_energy_only_inputs_sig =
        impl->model->signatures.at(COMPUTE_ENERGY_ONLY_KEY).inputs;
  }

  this->DEFAULT_INPUT_PREFIX = this->compute_function_name + "_";
  has_map_atoms_to_structure_op =
      impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "map_atoms_to_structure");
  has_nstruct_total_op = impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "n_struct_total");
  has_mu_i_op = impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "mu_i");
  has_batch_tot_nat = impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "batch_tot_nat");
  has_atomic_mu_i_local = impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "atomic_mu_i_local");

  // -- ALCHEMY: does this model expose the lambda-aware inputs? --
  has_lambda_input = impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "lambda_i") &&
                     impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "mu_a_i") &&
                     impl->model->has_graph_input(DEFAULT_INPUT_PREFIX + "mu_b_i");
  if (!has_lambda_input)
    error->all(FLERR, "pair grace/1layer/alch requires a lambda-aware exported model "
                      "(inputs lambda_i, mu_a_i, mu_b_i). Re-export with the alch head.");
  // resolve the alchemical A/B element indices from the model's element map
  if (!elements_to_index_map.count(element_A) || !elements_to_index_map.count(element_B))
    error->all(FLERR, "pair_style grace/1layer/alch elements '{}'/'{}' not in the model",
               element_A, element_B);
  index_A = elements_to_index_map.at(element_A);
  index_B = elements_to_index_map.at(element_B);
  if (comm->me == 0)
    utils::logmesg(lmp, "[GRACE/1L/alch] lambda morphs {} (mu={}) <-> {} (mu={})\n",
                   element_A, index_A, element_B, index_B);
}

/* ---------------------------------------------------------------------- */

void PairGRACE1LayerAlch::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style grace/1layer/chunk requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style grace/1layer/chunk requires newton pair on");

  neighbor->add_request(this, NeighConst::REQ_FULL);

  if (atom->map_style == Atom::MAP_NONE) {
    atom->map_init();
    atom->map_set();
  }

  // -- ALCHEMY: per-atom lambda from fix property/atom d_lambda ghost yes --
  int flag, cols, ghost;
  index_lambda = atom->find_custom_ghost("lambda", flag, cols, ghost);
  if (index_lambda < 0 || flag != 1 || cols != 0)
    error->all(FLERR, "pair grace/1layer/alch requires a per-atom scalar 'lambda': "
                      "fix <id> all property/atom d_lambda ghost yes");
  if (!ghost)
    error->all(FLERR, "fix property/atom d_lambda must use 'ghost yes'");

  // allocate dedlam now so fix alchemical/switch can resolve extract_peratom at init
  if (atom->nmax > nmax_dedlam) {
    memory->destroy(dedlam);
    nmax_dedlam = atom->nmax;
    memory->create(dedlam, nmax_dedlam, "pair:dedlam");
  }
}

/* ---------------------------------------------------------------------- */

double PairGRACE1LayerAlch::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  scale[j][i] = scale[i][j];
  if (is_custom_cutoffs) return cutoff_matrix_per_lammps_type[i][j];
  return cutoff;
}

/* ---------------------------------------------------------------------- */

void *PairGRACE1LayerAlch::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "compute_energy_only") == 0) return (void *) &flag_compute_energy_only;
  dim = 2;
  if (strcmp(str, "scale") == 0) return (void *) scale;
  return nullptr;
}

/* ---------------------------------------------------------------------- */

void PairGRACE1LayerAlch::compute(int eflag, int vflag)
{
  total_timer.start_step();
  data_timer.start_step();
  model_timer.start_step();
  tp_timer.start_step();

  total_timer.start();
  current_step_real_atoms = 0;
  impl->graph_recompiled = false;
  ev_init(eflag, vflag);

  int inum = list->inum;
  // Early exit if no local atoms to process (e.g., vacuum region)
  if (inum == 0) {
    total_timer.stop();
    return;
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // -- ALCHEMY: per-atom lambda (locals + ghosts) and dedlam accumulator --
  double *lambda = atom->dvector[index_lambda];
  if (atom->nmax > nmax_dedlam) {
    memory->destroy(dedlam);
    nmax_dedlam = atom->nmax;
    memory->create(dedlam, nmax_dedlam, "pair:dedlam");
  }
  for (int i = 0; i < nall; i++) dedlam[i] = 0.0;

  bool do_energy_only = flag_compute_energy_only && !debug_no_energy_only_calc;
  auto compute_inputs_sig = impl->compute_inputs_sig;
  if (do_energy_only) {
    if (has_compute_energy_only) {
      compute_inputs_sig = impl->compute_energy_only_inputs_sig;
    } else {
      do_energy_only = false;
      if (!warning_compute_energy_only_not_avail_shown) {
        utils::logmesg(
            lmp,
            std::string(
                "[GRACE:WARNING] Compute energy only function is not available, but requested. ") +
                "Full compute function will be used. This message is shown only once\n");
        warning_compute_energy_only_not_avail_shown = true;
      }
    }
  }

  // Initialize/resize global-to-chunk map
  if (impl->global_to_chunk_map.size() < (size_t) nall) {
    impl->global_to_chunk_map.assign(nall, -1);
  }

  int chunk_offset = 0;
  int chunk_idx = 0;

  while (chunk_offset < inum) {
    int current_chunk_size = std::min(chunksize, inum - chunk_offset);
    data_timer.start();

    // -- Phase 1: Topology Discovery --
    int node_counter = 0;
    impl->chunk_to_global_map.clear();

    // Map real atoms in chunk
    for (int ii = 0; ii < current_chunk_size; ++ii) {
      int i = ilist[chunk_offset + ii];
      impl->global_to_chunk_map[i] = node_counter;
      impl->chunk_to_global_map.push_back(i);
      node_counter++;
    }
    int n_real = node_counter;

    // Scan neighbors to discover pseudo-ghosts and bonds
    int n_bonds_real = 0;
    for (int ii = 0; ii < current_chunk_size; ++ii) {
      int i = ilist[chunk_offset + ii];
      int i_type = type[i];
      int i_neigh_num = numneigh[i];
      int *i_neigh_list = firstneigh[i];
      double xtmp = x[i][0];
      double ytmp = x[i][1];
      double ztmp = x[i][2];

      for (int jj = 0; jj < i_neigh_num; ++jj) {
        int j = i_neigh_list[jj] & NEIGHMASK;
        int j_type = type[j];

        double delx = xtmp - x[j][0];
        double dely = ytmp - x[j][1];
        double delz = ztmp - x[j][2];
        double rsq = delx * delx + dely * dely + delz * delz;
        double cutsq_ij = is_custom_cutoffs ? cutoff_matrix_per_lammps_type[i_type][j_type] *
                cutoff_matrix_per_lammps_type[i_type][j_type]
                                            : cutoff * cutoff;

        if (rsq < cutsq_ij) {
          if (impl->global_to_chunk_map[j] == -1) {
            impl->global_to_chunk_map[j] = node_counter;
            impl->chunk_to_global_map.push_back(j);
            node_counter++;
          }
          n_bonds_real++;
        }
      }
    }
    int n_nodes_in_chunk = node_counter;

    // Padding
    int n_nodes_padded = impl->atom_padding.update(n_nodes_in_chunk);
    if (impl->atom_padding.last_update_triggered_resize()) impl->graph_recompiled = true;
    int n_real_padded = impl->real_atom_padding.update(n_real);
    if (impl->real_atom_padding.last_update_triggered_resize()) impl->graph_recompiled = true;
    int n_bonds_padded = impl->neighbor_padding.update(n_bonds_real);
    if (impl->neighbor_padding.last_update_triggered_resize()) impl->graph_recompiled = true;
    current_step_real_atoms += current_chunk_size;

#ifdef GRACE_CHUNK_DEBUG
    {
      utils::logmesg(lmp, "[CHUNK-DBG] chunk_idx={}, offset={}, size={}\n", chunk_idx, chunk_offset,
                     current_chunk_size);
      utils::logmesg(lmp, "[CHUNK-DBG] n_nodes={} (n_real={}), n_padded={}, n_real_padded={}\n",
                     n_nodes_in_chunk, n_real, n_nodes_padded, n_real_padded);
      utils::logmesg(lmp, "[CHUNK-DBG] n_bonds_real={}, n_bonds_padded={}\n", n_bonds_real,
                     n_bonds_padded);
    }
#endif

    // -- Phase 2: Build Tensors --
    // Use element type 0 for padding slots (safe default for any valid model)
    // ALCHEMY: atomic_mu_i is sized to n_NODES (all chunk nodes incl pseudo-
    // ghosts), NOT n_real, because the lambda-aware export gathers per-NODE
    // blended rows (see [[grace-1L-2L-alchemy-design]]). lambda_node is the
    // matching per-node lambda; both filled from each node's global atom.
    impl->atomic_mu_i_local.assign(n_nodes_padded, 0);
    impl->lambda_i.assign(n_nodes_padded, 0.0);
    impl->ind_i.assign(n_bonds_padded, n_nodes_padded - 1);
    impl->ind_j.assign(n_bonds_padded, n_nodes_padded - 1);
    impl->mu_i.assign(n_bonds_padded, 0);
    impl->mu_j.assign(n_bonds_padded, 0);
    impl->bond_vector.assign(3 * n_bonds_padded, 1e6);
    impl->map_atoms_to_structure.assign(n_nodes_padded, 0);

    // Fill ALL nodes (real + pseudo-ghost) with element index + per-node lambda.
    // Ghost lambdas are valid because d_lambda is declared 'ghost yes'.
    for (int k = 0; k < n_nodes_in_chunk; ++k) {
      int g_idx = impl->chunk_to_global_map[k];
      impl->atomic_mu_i_local[k] = element_type_mapping[type[g_idx]];
      impl->lambda_i[k] = lambda[g_idx];
    }

    // Fill real bonds data
    int bond_idx = 0;
    for (int ii = 0; ii < current_chunk_size; ++ii) {
      int i = ilist[chunk_offset + ii];
      int i_type = type[i];
      int i_neigh_num = numneigh[i];
      int *i_neigh_list = firstneigh[i];
      int i_chunk = impl->global_to_chunk_map[i];

      for (int jj = 0; jj < i_neigh_num; ++jj) {
        int j = i_neigh_list[jj] & NEIGHMASK;
        int j_type = type[j];
        double dx = x[j][0] - x[i][0];
        double dy = x[j][1] - x[i][1];
        double dz = x[j][2] - x[i][2];
        double rsq = dx * dx + dy * dy + dz * dz;
        double cutsq_ij = is_custom_cutoffs ? cutoff_matrix_per_lammps_type[i_type][j_type] *
                cutoff_matrix_per_lammps_type[i_type][j_type]
                                            : cutoff * cutoff;

        if (rsq < cutsq_ij) {
          int j_chunk = impl->global_to_chunk_map[j];
          impl->ind_i[bond_idx] = i_chunk;
          impl->ind_j[bond_idx] = j_chunk;
          impl->mu_i[bond_idx] = element_type_mapping[type[i]];
          impl->mu_j[bond_idx] = element_type_mapping[type[j]];
          impl->bond_vector[3 * bond_idx + 0] = dx;
          impl->bond_vector[3 * bond_idx + 1] = dy;
          impl->bond_vector[3 * bond_idx + 2] = dz;
          bond_idx++;
        }
      }
    }

    std::vector<std::tuple<std::string, cppflow::tensor>> inputs;
    // ALCHEMY: atomic_mu_i sized n_nodes_padded (per-node element idx)
    inputs.emplace_back(compute_inputs_sig.at("atomic_mu_i").name,
                        cppflow::tensor(impl->atomic_mu_i_local, {n_nodes_padded}));
    // ALCHEMY: per-node lambda + scalar 'to' element B
    inputs.emplace_back(compute_inputs_sig.at("lambda_node").name,
                        cppflow::tensor(impl->lambda_i, {n_nodes_padded}));
    inputs.emplace_back(compute_inputs_sig.at("b_idx").name,
                        cppflow::tensor(std::vector<int32_t>{(int32_t) index_B}, {}));
    inputs.emplace_back(compute_inputs_sig.at("ind_i").name,
                        cppflow::tensor(impl->ind_i, {n_bonds_padded}));
    inputs.emplace_back(compute_inputs_sig.at("ind_j").name,
                        cppflow::tensor(impl->ind_j, {n_bonds_padded}));
    inputs.emplace_back(compute_inputs_sig.at("bond_vector").name,
                        cppflow::tensor(impl->bond_vector, {n_bonds_padded, 3}));
    inputs.emplace_back(compute_inputs_sig.at("batch_tot_nat_real").name,
                        cppflow::tensor(std::vector<int32_t>{n_real}, {}));
    inputs.emplace_back(compute_inputs_sig.at("batch_tot_nat").name,
                        cppflow::tensor(std::vector<int32_t>{n_nodes_padded}, {}));
    inputs.emplace_back(compute_inputs_sig.at("n_struct_total").name,
                        cppflow::tensor(std::vector<int32_t>{1}, {}));
    inputs.emplace_back(compute_inputs_sig.at("batch_total_num_structures").name,
                        cppflow::tensor(std::vector<int32_t>{1}, {}));
    inputs.emplace_back(compute_inputs_sig.at("n_neigh_real").name,
                        cppflow::tensor(std::vector<int32_t>{n_bonds_real}, {}));
    inputs.emplace_back(compute_inputs_sig.at("map_atoms_to_structure").name,
                        cppflow::tensor(impl->map_atoms_to_structure, {n_nodes_padded}));
    inputs.emplace_back(compute_inputs_sig.at("map_bonds_to_structure").name,
                        cppflow::tensor(std::vector<int32_t>(n_bonds_padded, 0), {n_bonds_padded}));

#ifdef GRACE_CHUNK_DEBUG
    GRACE::print_tf_inputs(inputs, comm->me, lmp, false);
#endif

    // ALCHEMY: alch model always returns atomic_energy, z_pair_f, dedlam
    // (no energy-only path for the alch style).
    std::vector<std::string> output_names;
    {
      auto sig_outputs = impl->model->signatures.at(compute_function_name).outputs;
      output_names.push_back(sig_outputs.at("atomic_energy").name);
      output_names.push_back(sig_outputs.at("z_pair_f").name);
      output_names.push_back(sig_outputs.at("dedlam").name);
    }
    data_timer.stop();

    model_timer.start();
    auto outputs = impl->model->operator()(inputs, output_names);
    model_timer.stop();

    data_timer.start();

    // We rely on TF model returning correct output sizes
    const double *e_data =
        static_cast<const double *>(TF_TensorData(outputs[0].get_tensor().get()));
    const double *f_data =
        static_cast<const double *>(TF_TensorData(outputs[1].get_tensor().get()));
    const double *dedlam_data =                 // ALCHEMY: per-node dE/dlambda
        static_cast<const double *>(TF_TensorData(outputs[2].get_tensor().get()));

    // ALCHEMY: scatter per-node dedlam back onto global atoms (locals+ghosts);
    // ghosts fold to owners via reverse_comm after the chunk loop.
    for (int k = 0; k < n_nodes_in_chunk; ++k) {
      int g_idx = impl->chunk_to_global_map[k];
      dedlam[g_idx] += dedlam_data[k];
    }

    // -- Phase 4: Scattering --
    bond_idx = 0;
    for (int ii = 0; ii < current_chunk_size; ++ii) {
      int i = ilist[chunk_offset + ii];
      int i_type = type[i];
      double i_scale = scale[i_type][i_type];

      // Energy
      if (eflag_either) {
        double evdwl = i_scale * e_data[ii];
        ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
      }

      if (!do_energy_only) {
        // Forces
        // z_pair_f maps to bonds
        int i_neigh_num = numneigh[i];
        int *i_neigh_list = firstneigh[i];
        int i_chunk = impl->global_to_chunk_map[i];

        for (int jj = 0; jj < i_neigh_num; ++jj) {
          int j = i_neigh_list[jj] & NEIGHMASK;
          int j_type = type[j];
          double rsq = 0.0;
          // recalculate rsq or store it? storing is memory intensive
          // Just use bond_idx sequence which matches the construction order

          double cutsq_ij = is_custom_cutoffs ? cutoff_matrix_per_lammps_type[i_type][j_type] *
                  cutoff_matrix_per_lammps_type[i_type][j_type]
                                              : cutoff * cutoff;

          // Re-calculate distance to check cutoff and increment bond_idx
          // This is slightly inefficient but robust given existing code structure
          // and avoids large index mapping arrays
          double dx = x[j][0] - x[i][0];
          double dy = x[j][1] - x[i][1];
          double dz = x[j][2] - x[i][2];
          rsq = dx * dx + dy * dy + dz * dz;

          if (rsq < cutsq_ij) {
            // Access force from f_data [bond_idx, 3]
            // Note: f_data contains gradients, so force is negative gradient
            double fx = -i_scale * f_data[3 * bond_idx + 0];
            double fy = -i_scale * f_data[3 * bond_idx + 1];
            double fz = -i_scale * f_data[3 * bond_idx + 2];

            // Apply to i
            f[i][0] += fx;
            f[i][1] += fy;
            f[i][2] += fz;

            // Apply reaction force to j. newton_pair is always ON for this pair style
            // (enforced in init_style), so we always apply forces to both atoms.
            f[j][0] -= fx;
            f[j][1] -= fy;
            f[j][2] -= fz;

            // Virial: ev_tally_xyz expects delx = x[i]-x[j] (LAMMPS convention); dx = x[j]-x[i] already computed above
            if (vflag_either || vflag_global) {
              ev_tally_xyz(i, j, nlocal, force->newton_pair, 0.0, 0.0, fx, fy, fz, -dx, -dy, -dz);
            }
            if (cvflag_atom) {
              cvatom[i][0] += 0.5 * (-dx) * fx;    // xx
              cvatom[i][1] += 0.5 * (-dy) * fy;    // yy
              cvatom[i][2] += 0.5 * (-dz) * fz;    // zz
              cvatom[i][3] += 0.5 * (-dx) * fy;    // xy
              cvatom[i][4] += 0.5 * (-dx) * fz;    // xz
              cvatom[i][5] += 0.5 * (-dy) * fz;    // yz
              cvatom[i][6] += 0.5 * (-dy) * fx;    // yx
              cvatom[i][7] += 0.5 * (-dz) * fx;    // zx
              cvatom[i][8] += 0.5 * (-dz) * fy;    // zy

              cvatom[j][0] += 0.5 * (-dx) * fx;
              cvatom[j][1] += 0.5 * (-dy) * fy;
              cvatom[j][2] += 0.5 * (-dz) * fz;
              cvatom[j][3] += 0.5 * (-dx) * fy;
              cvatom[j][4] += 0.5 * (-dx) * fz;
              cvatom[j][5] += 0.5 * (-dy) * fz;
              cvatom[j][6] += 0.5 * (-dy) * fx;
              cvatom[j][7] += 0.5 * (-dz) * fx;
              cvatom[j][8] += 0.5 * (-dz) * fy;
            }

            bond_idx++;
          }
        }
      }
    }

    // -- Phase 5: Cleanup --
    for (int k = 0; k < n_nodes_in_chunk; ++k) {
      impl->global_to_chunk_map[impl->chunk_to_global_map[k]] = -1;
    }

    chunk_offset += current_chunk_size;
    chunk_idx++;
  }

  // ALCHEMY: fold ghost dedlam contributions back onto owning atoms
  comm->reverse_comm(this);

  total_timer.stop();

  if (impl->graph_recompiled) {
    total_timer.rollback();
    data_timer.rollback();
    model_timer.rollback();
    tp_timer.rollback();
  } else {
    total_timer.commit();
    data_timer.commit();
    model_timer.commit();
    tp_timer.commit();
    total_real_atoms_processed += current_step_real_atoms;
    total_compute_calls++;
  }

#ifdef GRACE_PROFILE
  if (inum > 0) {
    double d_t = data_timer.as_microseconds();
    double m_t = model_timer.as_microseconds();
    double total_t = total_timer.as_microseconds();
    auto pct = [&](double t) {
      return (total_t > 0) ? (t / total_t * 100.0) : 0.0;
    };

    fprintf(stderr,
            "[GRACE-PROFILE] [Rank %d] Timings (mcs): Data: %.1f (%.1f%%) | Model: %.1f (%.1f%%) | "
            "Total: %.1f\n",
            comm->me, d_t, pct(d_t), m_t, pct(m_t), total_t);
  }
#endif
}

/* ----------------------------------------------------------------------
   ALCHEMY: reverse-comm of per-atom dedlam (ghost -> owner accumulation)
------------------------------------------------------------------------- */

int PairGRACE1LayerAlch::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) buf[m++] = dedlam[i];
  return m;
}

void PairGRACE1LayerAlch::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) dedlam[list[i]] += buf[m++];
}

/* ----------------------------------------------------------------------
   per-atom access: dedlam[i] = dE_total/dlambda_i (eV). Matches the
   grace/fs/alch contract so `fix alchemical/switch` consumes it unchanged.
------------------------------------------------------------------------- */

void *PairGRACE1LayerAlch::extract_peratom(const char *str, int &ncol)
{
  if (strcmp(str, "dedlam") == 0) {
    ncol = 0;
    return (void *) dedlam;
  }
  return nullptr;
}

#endif
