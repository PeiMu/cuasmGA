# GA Strategy Implementations

Historical implementations of the five mutation and crossover strategies of the project report. Each file is a self-contained
version of `newga.py` from a different development phase.

## Files

### newga_phase1_nonadj_swap.py (Report §4.1.1–4.1.2)
Adjacent swap and non-adjacent swap mutation with transitive dependency
checking. Crossover is identity (returns parent copies). Includes register
expansion for HMMA/LDSM wide-register dependencies.

### newga_phase2_barrier_crossover.py (Report §4.2.1)
Adds `parse_barrier_groups()` which segments SASS by BAR.SYNC positions.
Barrier-group crossover was tested but abandoned: >95% of offspring failed
correctness due to register liveness spanning BAR.SYNC boundaries. The
crossover function was reverted to identity; the parsing function remains.

### newga_phase3_hmma_ox_crossover.py (Report §4.1.3, §4.2.2)
Complete implementation with all five strategies:
- **Adjacent swap** (§4.1.1): single-position memory instruction swap
- **Non-adjacent swap** (§4.1.2): distance 2–20, transitive dependency check
- **HMMA sub-block mutation** (§4.1.3): swap within consecutive HMMA runs
- **HMMA-memory interleave mutation**: insert memory ops into HMMA clusters
- **Cluster mutation** (disabled, `if False:`): FADD/FSETP permutation caused CUDA crashes
- **HMMA permutation crossover with OX** (§4.2.2): Order Crossover on HMMA sub-block orderings
- **Barrier-group crossover** (§4.2.1): fallback path, reverted to identity
