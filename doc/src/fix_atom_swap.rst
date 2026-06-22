.. index:: fix atom/swap

fix atom/swap command
=====================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID atom/swap N X seed T keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* atom/swap = style name of this fix command
* N = invoke this fix every N steps
* X = number of swaps to attempt every N steps
* seed = random # seed (positive integer)
* T = scaling temperature of the MC swaps (temperature units, or equal-style variable v_name)
* one or more keyword/value pairs may be appended to args
* keyword = *types* or *mu* or *ke* or *semi-grand* or *region* or *swap_count* or *noforce* or *localE* or *adapt*

  .. parsed-literal::

       *types* values = two or more atom types (1-Ntypes or type label)
       *mu* values = chemical potential of swap types (energy units, or equal-style variable v_name)
       *ke* value = *no* or *yes*
         *no* = no conservation of kinetic energy after atom swaps
         *yes* = kinetic energy is conserved after atom swaps
       *semi-grand* value = *no* or *yes*
         *no* = particle type counts and fractions conserved
         *yes* = semi-grand canonical ensemble, particle fractions not conserved
       *region* value = region-ID
         region-ID = ID of region to use as an exchange/move volume
       *swap_count* value = N
         N = number of atom pairs to include in each MC move (positive integer)
       *noforce* value = *no* or *yes*
         *no* = compute full energy and forces during MC energy evaluation
         *yes* = skip force accumulation during MC energy evaluation (energy only)
       *localE* value = *no* or *yes*
         *no* = evaluate full system energy for each MC trial
         *yes* = evaluate only the local coordination-shell energy change (PACE only)
       *adapt* values = dX K [*maxdmu* value] [*tracked* N] [*mumin* value] [*mumax* value]
         dX = target composition step per mu update (dimensionless, in (0,1))
         K = number of MC blocks to accumulate before each mu update
         *maxdmu* value = maximum absolute mu step per update (energy units, default 10.0)
         *tracked* value = 1 or 2 — which of the two listed types has its mu driven (default 2)
         *mumin* value = hard lower bound on the adaptive mu value (energy units, default: none)
         *mumax* value = hard upper bound on the adaptive mu value (energy units, default: none)

Examples
""""""""

.. code-block:: LAMMPS

   fix 2 all atom/swap 1 1 29494 300.0 ke no types 1 2
   fix myFix all atom/swap 100 1 12345 298.0 region my_swap_region types 5 6
   fix SGMC all atom/swap 1 100 345 1.0 semi-grand yes types 1 2 3 mu 0.0 4.3 -5.0
   fix adaptSGMC all atom/swap 10 200 345 1.0 semi-grand yes types 1 2 mu 0.0 -3.0 adapt 0.02 10
   fix adaptSGMC all atom/swap 10 200 345 1.0 semi-grand yes types 1 2 mu 0.0 -3.0 adapt 0.02 10 maxdmu 2.0 mumin -15.0 mumax 5.0
   fix multiSwap all atom/swap 1 1 29494 300.0 types 1 2 swap_count 5   fix fastSwap all atom/swap 1 1 29494 300.0 types 1 2 noforce yes
   fix localSwap all atom/swap 100 100 12 800 ke no types 1 2 localE yes

Description
"""""""""""

This fix performs Monte Carlo swaps of atoms of one given atom type with
atoms of the other given atom types.  The specified scaling temperature
*T* is used in the Metropolis criterion dictating swap probabilities.
*T* may be given as a numeric constant or as an equal-style variable
using the ``v_name`` syntax, in which case it is re-evaluated once per
MC block (i.e., once every *N* MD steps).  This allows temperature
schedules such as ``ramp(T_start, T_end)`` to be applied to the MC
acceptance criterion independently of the MD thermostat temperature.
The variable value must be positive at every evaluation.

.. versionchanged:: TBD

Perform *X* swaps of atoms of one type with atoms of another type
according to a Monte Carlo probability.  Swap candidates must be in
the fix group, must be in the region (if specified), and must be of
one of the listed types. Swaps are attempted between candidates that
are chosen randomly with equal probability among the candidate
atoms. Swaps are not attempted between atoms of the same type since
nothing would happen.

All atoms in the simulation domain can also be moved using regular
time integration displacements (e.g., via :doc:`fix nvt <fix_nh>`),
resulting in a hybrid MC+MD simulation, where $X$ MC swap attempts are
made once every $N$ MD steps.  A smaller-than-usual timestep size may
be needed when running such a hybrid simulation, especially if the
swapped atoms are not well equilibrated.

.. note::

   To run an MC-only simulation (no MD), you should define no
   time-integration fix, set the :doc:`thermo <thermo>` command to 1,
   set *N* to 1, and set *X* small enough to see the MC evolution of
   the system.  But if *X* is too small, the overhead at the start and
   stop of MC moves each timestep will slow down the simulation.

The *types* keyword is required. At least two atom types must be
specified. If not using *semi-grand*, exactly two atom types are
required.

The *ke* keyword can be set to *no* to turn off kinetic energy
conservation for swaps. The default is *yes*, which means that swapped
atoms have their velocities scaled by the ratio of the masses of the
swapped atom types. This ensures that the kinetic energy of each atom is
the same after the swap as it was before the swap, even though the atom
masses have changed.

The *semi-grand* keyword can be set to *yes* to switch to the semi-grand
canonical ensemble as discussed in :ref:`(Sadigh) <Sadigh>`. This means
that the total number of each particle type does not need to be
conserved. The default is *no*, which means that the only kind of swap
allowed exchanges an atom of one type with an atom of a different given
type. In other words, the relative mole fractions of the swapped atoms
remains constant. Whereas in the semi-grand canonical ensemble, the
composition of the system can change. Note that when using *semi-grand*,
atoms in the fix group whose type is not listed in the *types* keyword
are ineligible for attempted conversion. An attempt is made to switch
the selected atom (if eligible) to one of the other listed types with
equal probability.  Acceptance of each attempt depends upon the
Metropolis criterion.

The *mu* keyword allows users to specify chemical potentials. This is
required and allowed only when using *semi-grand*\ .  All chemical
potentials are absolute, so there is one for each swap type listed
following the *types* keyword.  In semi-grand canonical ensemble
simulations the chemical composition of the system is controlled by the
difference in these values. So shifting all values by a constant amount
will have no effect on the simulation.

Each *mu* value may be given either as a numeric constant or as an
equal-style variable using the ``v_name`` syntax.  Variable-style mu
values are re-evaluated once per MC block (i.e., once every *N* MD
steps), so they track time-dependent schedules such as ``ramp()``,
``v_othervar``, or any equal-style expression.  Because ``ramp(v1,v2)``
evaluates at the current timestep, and the MC block fires at timesteps
:math:`t_0,\ t_0 + N,\ t_0 + 2N, \ldots`, the chemical potential is
sampled exactly on those timesteps and the ramp is traversed uniformly.
For example:

.. code-block:: LAMMPS

   variable dmu equal ramp(-5.0, 5.0)
   fix SGMC all atom/swap 10 100 345 1.0 semi-grand yes types 1 2 mu v_dmu 0.0
   run 10000   # 1000 distinct mu values, evenly spaced from -5 to 5

Mixing constants and variables is allowed.

.. versionchanged:: TBD

This command may optionally use the *region* keyword to define swap
volume.  The specified region must have been previously defined with a
:doc:`region <region>` command.  It must be defined with side = *in*\ .
Swap attempts occur only between atoms that are both within the
specified region. Swaps are not otherwise attempted.

.. versionchanged:: TBD

The *swap_count* keyword sets the number of atom pairs that are swapped
atomically in a single MC move.  The default value of 1 reproduces the
original single-pair behavior.  When *swap_count* is set to *N* > 1,
each MC trial selects *N* distinct atoms of the first swap type and *N*
distinct atoms of the second swap type and swaps all of them
simultaneously before evaluating the total system energy.  The move is
the accepted or rejected as a whole based on the Metropolis criterion
applied to the total energy change.  This is not compatible with
*semi-grand*.  The number of eligible atoms of each swap type must be
at least *N*; if not, the swap attempt is silently skipped.

.. versionchanged:: TBD

The *noforce* keyword controls whether force accumulation is skipped
during the MC energy evaluation calls.  When set to *yes*, the pair
potential is called with an internal ``energy_only`` flag that bypasses
the force derivative loop, reducing the cost of each MC trial.  Only
the energy is computed; the ``neighbours_forces`` array is not
populated.  This is safe because forces are fully recomputed at the
next regular MD timestep.  The speedup is largest for ML potentials
(e.g. ML-PACE/ACE) where the force loop dominates compute time.  Not
all pair styles support this flag; styles that do not will simply
ignore it and compute forces as normal.

.. versionchanged:: TBD

The *localE* keyword activates a local coordination-shell energy
approximation for the MC acceptance criterion.  When set to *yes*,
the fix maintains a cached array of per-atom ACE energies (built once
per MD step after reneighboring) and evaluates the energy change of a
swap trial by recomputing only the energy of the two swapped atoms and
their first coordination shells (typically ~50-100 atoms), rather than
performing a full system-wide pair compute.  The energy change is exact
within the ACE formalism because each atom's energy depends only on its
own neighbour shell.  The cache is updated incrementally on accepted
swaps and rebuilt from scratch at the start of each MC block.

This option is currently supported only with *pair_style pace* and
requires *swap_count* = 1.  It is not compatible with unequal pair
cutoffs between the swap types.  On parallel runs, a single
``MPI_Allreduce`` per trial replaces the full pair compute reduction.
The expected speedup over *noforce yes* is proportional to N / Z, where
N is the number of atoms and Z is the coordination number (typically
50-80x for a 4000-atom system).

The *localE* optimization works in both the regular (composition-conserving
pair-swap) mode and in *semi-grand* = yes mode.  In semi-grand mode each
trial changes the type of a single atom, and the local shell delta is
combined with the chemical-potential term :math:`\mu_j - \mu_i` exactly as
in the full-energy path, so the accept/reject statistics are unchanged --
only the energy evaluation is accelerated.  In semi-grand mode *localE*
requires a single ``pair_style pace`` (the hybrid/scaled split-cache path
is only available in the regular pair-swap mode).

.. versionadded:: TBD

The *adapt* keyword activates susceptibility-driven adaptive stepping of
the chemical potential (``mu``) for the tracked swap type.  It is only
compatible with *semi-grand yes* and exactly two swap types.  Rather than
traversing a pre-defined mu schedule at equal spacing, the fix measures
the local thermodynamic susceptibility

.. math::

   \chi \equiv \frac{d\langle X \rangle}{d\mu} = \frac{\beta}{N} \mathrm{Var}(N_2)

from the fluctuations of the tracked-type count accumulated over
*K* consecutive MC blocks.  It then updates mu by

.. math::

   \Delta\mu = \frac{\Delta X_\mathrm{target}}{\chi}

so that each update moves the expected composition by *dX*.  This
automatically concentrates mu points near steep transitions
(large :math:`\chi`) and takes large steps in flat regions (small
:math:`\chi`), giving uniform resolution in composition space without
any user knowledge of where the transition occurs.

By default the second of the two listed types is the adapted species
(i.e., its ``mu`` entry is driven).  Use the *tracked* sub-keyword to
change which type is tracked: ``tracked 1`` drives the mu of the first
listed type instead.  Note that the initial value of the adaptive mu is
taken from the ``mu`` value supplied for that species, so it should be
set to the desired starting chemical potential.

The optional *maxdmu* sub-keyword (default: 10.0 in energy units) caps
:math:`|\Delta\mu|` per step, preventing runaway in regions where
composition is truly flat and :math:`\chi \approx 0`.

The optional *mumin* and *mumax* sub-keywords set hard lower and upper
bounds on the adaptive mu value.  Once the running mu reaches a bound it
is clamped there and will not cross it regardless of the susceptibility
estimate.  These are useful to confine the scan to a physically
relevant range and to prevent runaway at the edges of a phase diagram.

The current composition :math:`X` (tracked-type fraction), local
susceptibility :math:`\chi`, and current adaptive mu value are
accessible as ``f_ID[3]``, ``f_ID[4]``, and ``f_ID[5]`` respectively,
and can be monitored via :doc:`thermo_style custom <thermo_style>`.

The *adapt* keyword is not compatible with variable-backed mu (``v_``
syntax) for the driven type, nor with ``swap_count > 1``.

You should ensure you do not swap atoms belonging to a molecule, or
LAMMPS will eventually generate an error when it tries to find those
atoms.  LAMMPS will warn you if any of the atoms eligible for swapping
have a non-zero molecule ID, but does not check for this at the time of
swapping.

If not using *semi-grand* this fix checks to ensure all atoms of the
given types have the same atomic charge. LAMMPS does not enforce this in
general, but it is needed for this fix to simplify the swapping
procedure. Successful swaps will swap the atom type and charge of the
swapped atoms. Conversely, when using *semi-grand*, it is assumed that
all the atom types involved in switches have the same charge. Otherwise,
charge would not be conserved. As a consequence, no checks on atomic
charges are performed, and successful switches update the atom type but
not the atom charge. While it is possible to use *semi-grand* with
groups of atoms that have different charges, these charges will not be
changed when the atom types change.  The same applies for systems
with per-atom masses: non *semi-grand* will swap atom masses, but
the masses have to be the same each for the atom types.  When using
*semi-grand* no per-atom masses are changed.

Since this fix computes total potential energies before and after
proposed swaps, even complicated potential energy calculations are
acceptable, including the following:

* long-range electrostatics (:math:`k`-space)
* many body pair styles
* hybrid pair styles (with restrictions)
* EAM pair styles
* triclinic systems

Some fixes have an associated potential energy. Examples of such fixes
include: :doc:`efield <fix_efield>`, :doc:`gravity <fix_gravity>`,
:doc:`addforce <fix_addforce>`, :doc:`langevin <fix_langevin>`,
:doc:`restrain <fix_restrain>`, :doc:`temp/berendsen
<fix_temp_berendsen>`, :doc:`temp/rescale <fix_temp_rescale>`, and
:doc:`wall fixes <fix_wall>`.  For that energy to be included in the
total potential energy of the system (the quantity used when
performing GCMC moves), you **must** enable the :doc:`fix_modify
<fix_modify>` *energy* option for that fix.  The doc pages for
individual :doc:`fix <fix>` commands specify if this should be done.

----------

Dump image info
"""""""""""""""

.. versionadded:: 11Feb2026

Fix *atom/swap* supports the *fix* keyword of :doc:`dump image
<dump_image>`.  The fix will pass geometry information about atoms
involved in a swap to *dump image* so that these atoms can be
highlighted in the visualization as additional spheres.  For how
long those additional spheres will be shown depends on the value of the
*vizsteps* setting (default is 1000) which can be changed by using the
:doc:`fix_modify command <fix_modify>`.  If an atom is involved in
multiple swaps, the check on showing the additional graphics depends
on the timestep of its last swap.

The color of the additional spheres is by default that of the atom type
*before* the swap when using color styles "type" or "element".  With
color style "const" the default value of "white" can be changed using
:doc:`dump_modify fcolor <dump_image>`.  The transparency is by default
fully opaque and can be changed with *dump\_modify ftrans*\ .

The *fflag1* setting of *dump image fix* has no effect.

The *fflag2* setting allows you to set the radius of the added
spheres, since the radius is set to zero internally.

----------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix writes the state of the fix to :doc:`binary restart files
<restart>`.  This includes information about the random number generator
seed, the next timestep for MC exchanges, the number of exchange
attempts and successes, etc.  See the :doc:`read_restart <read_restart>`
command for info on how to re-specify a fix in an input script that
reads a restart file, so that the operation of the fix continues in an
uninterrupted fashion.

.. note::

   For this to work correctly, the timestep must **not** be changed
   after reading the restart with :doc:`reset_timestep
   <reset_timestep>`.  The fix will try to detect it and stop with an
   error.

None of the :doc:`fix_modify <fix_modify>` options are relevant to this
fix.

This fix computes a global vector of length 5, which can be accessed
by various :doc:`output commands <Howto_output>`.  The vector values are
the following global quantities:

  #. swap attempts (cumulative)
  #. swap accepts (cumulative)
  #. current composition :math:`X` of the tracked type (0 when *adapt* is not active)
  #. current susceptibility :math:`\chi = \beta\,\mathrm{Var}(N_2)/N` (0 when *adapt* is not active)
  #. current value of the adaptive mu (0 when *adapt* is not active)

The vector values calculated by this fix are "intensive".

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
doc page for more info.

When this fix is used with a :doc:`hybrid pair style <pair_hybrid>`
system, only swaps between atom types of the same sub-style (or
combination of sub-styles) are permitted.

This fix can be used with systems that have per-atom masses
(e.g. atom style sphere) provided all atoms of the types handled
by this fix have the same mass per type. The fix will check for that.
In case both, per-type and per-atom masses are present, a warning is printed.

Related commands
""""""""""""""""

:doc:`fix nvt <fix_nh>`, :doc:`neighbor <neighbor>`,
:doc:`fix deposit <fix_deposit>`, :doc:`fix evaporate <fix_evaporate>`,
:doc:`delete_atoms <delete_atoms>`, :doc:`fix gcmc <fix_gcmc>`,
:doc:`fix mol/swap <fix_mol_swap>`, :doc:`fix sgcmc <fix_sgcmc>`

Default
"""""""

The option defaults are *ke* = yes, *semi-grand* = no, *mu* = 0.0 for
all atom types, *swap_count* = 1, *noforce* = no, *localE* = no.

----------

.. _Sadigh:

**(Sadigh)** B Sadigh, P Erhart, A Stukowski, A Caro, E Martinez, and
L Zepeda-Ruiz, Phys. Rev. B, 85, 184203 (2012).
