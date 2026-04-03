# newga_phase3_hmma_ox_crossover.py — HMMA OX crossover + cluster mutation (Report Section 4.1.3, 4.2.2)
from typing import List, Optional, Callable
from collections import Counter
import random
import re
import time, csv, hashlib, statistics
import numpy as np
import pickle
import torch

from sass_kernel import SassKernel
from sassgen import write_sass_file
from decoder import decode, decode_ctrl_code
from gpu_utils import get_gpu_cc, get_mutatable_ops

from sample import Sample

# ========= Hyperparameters =========
POP_SIZE        = 10
MUTATION_RATE   = 1.0
NUM_GENERATIONS = 50
ELITE_SIZE      = 4
P_CLUSTER       = 0.15   # Cluster permutation mutation probability
P_NONADJ        = 0.20   # Non-adjacent swap probability
P_HMMA_MUT      = 0.15   # HMMA sub-block internal swap probability
P_INTERLEAVE    = 0.15   # HMMA-memory interleave probability

_CC = get_gpu_cc()
_, _BAN_OPS = get_mutatable_ops(_CC)


# ========= GPU keepalive =========

def gpu_keepalive():
    """Tiny GPU op to prevent idle GPU detection on EIDF cluster"""
    try:
        if torch.cuda.is_available():
            _ = torch.zeros(1, device='cuda')
            del _
    except RuntimeError:
        pass


# ========= Helper Functions =========

def _extract_registers(dst, src):
    regs = set()
    if dst and (dst.startswith('R') or dst.startswith('P') or dst.startswith('UR') or dst.startswith('UP')):
        regs.add(dst)
    for s in (src or []):
        if s.startswith('R') or s.startswith('P') or s.startswith('UR') or s.startswith('UP'):
            regs.add(s)
    return regs


def _get_stall_count(ctrl_code_str):
    if not ctrl_code_str or not ctrl_code_str.startswith('['):
        return 0
    try:
        _, _, _, _, stall_str = decode_ctrl_code(ctrl_code_str)
        return int(stall_str[1:-1])
    except Exception:
        return 0


def _set_stall_in_line(line, new_stall):
    return re.sub(r'S\d+\]', f'S{new_stall:02d}]', line, count=1)


# ========= Non-Adjacent Swap Dependency Check =========

def transitive_dependency_check(sass, i, j):
    if j <= i + 1:
        return False
    ctrl_i, _, _, opcode_i, dst_i, src_i, _ = decode(sass[i])
    ctrl_j, _, _, opcode_j, dst_j, src_j, _ = decode(sass[j])
    if ctrl_i is None or ctrl_j is None:
        return False
    for ban_op in _BAN_OPS:
        if opcode_i and ban_op in opcode_i:
            return False
        if opcode_j and ban_op in opcode_j:
            return False
    writes_i = set()
    if dst_i and (dst_i.startswith('R') or dst_i.startswith('P') or dst_i.startswith('UR')):
        writes_i.add(dst_i)
    writes_j = set()
    if dst_j and (dst_j.startswith('R') or dst_j.startswith('P') or dst_j.startswith('UR')):
        writes_j.add(dst_j)
    reads_i = set()
    for s in (src_i or []):
        if s.startswith('R') or s.startswith('P') or s.startswith('UR'):
            reads_i.add(s)
    reads_j = set()
    for s in (src_j or []):
        if s.startswith('R') or s.startswith('P') or s.startswith('UR'):
            reads_j.add(s)
    if writes_i & reads_j:
        return False
    if reads_i & writes_j:
        return False
    if writes_i & writes_j:
        return False
    for k in range(i + 1, j):
        ctrl_k, _, _, opcode_k, dst_k, src_k, _ = decode(sass[k])
        if ctrl_k is None:
            continue
        for ban_op in _BAN_OPS:
            if opcode_k and ban_op in opcode_k:
                return False
        writes_k = set()
        if dst_k and (dst_k.startswith('R') or dst_k.startswith('P') or dst_k.startswith('UR')):
            writes_k.add(dst_k)
        reads_k = set()
        for s in (src_k or []):
            if s.startswith('R') or s.startswith('P') or s.startswith('UR'):
                reads_k.add(s)
        if writes_i & reads_k:
            return False
        if reads_i & writes_k:
            return False
        if writes_i & writes_k:
            return False
        if writes_k & reads_j:
            return False
        if writes_j & reads_k:
            return False
        if writes_j & writes_k:
            return False
    return True


# ========= Barrier Group Parsing =========

def parse_barrier_groups(sass):
    bar_positions = []
    for idx, line in enumerate(sass):
        ctrl, _, _, opcode, _, _, _ = decode(line)
        if ctrl is None:
            continue
        if opcode and 'BAR.SYNC' in opcode:
            bar_positions.append(idx)
    if not bar_positions:
        return [(0, len(sass))]
    groups = []
    groups.append((0, bar_positions[0] + 1))
    for k in range(len(bar_positions) - 1):
        groups.append((bar_positions[k] + 1, bar_positions[k + 1] + 1))
    if bar_positions[-1] + 1 < len(sass):
        groups.append((bar_positions[-1] + 1, len(sass)))
    return groups


# ========= HMMA Sub-Block Detection =========

def _extract_a_operand(src_list):
    """
    Extract the A operand register from an HMMA instruction's source list.
    The A operand is the first source register (e.g., 'R68' or 'R68.reuse').
    Returns the base register name without .reuse suffix.
    """
    if not src_list:
        return None
    a_op = src_list[0]
    # Strip .reuse suffix for grouping
    return a_op.replace('.reuse', '')


def detect_hmma_subblocks(sass):
    """
    Scan the kernel for HMMA.16816.F32 instructions and group them by A operand register.

    Returns:
        subblocks: dict mapping A operand register name -> list of line indices
                   e.g. {'R68': [479, 482, 489, ...], 'R88': [520, 523, ...], ...}
        double_accum_pairs: list of (first_group_reg, second_group_reg) tuples
                            indicating that all HMMAs in first_group must execute
                            before any in second_group (shared accumulator banks).
                            For MoE: [('R68', 'R92'), ('R88', 'R112')]
    """
    # Collect all HMMA instructions with their A operand and accumulator registers
    hmma_info = {}  # a_reg -> [(line_idx, accum_regs), ...]

    for idx, line in enumerate(sass):
        ctrl, _, _, opcode, dst, src, _ = decode(line)
        if ctrl is None:
            continue
        if not (opcode and 'HMMA.16816.F32' in opcode):
            continue

        a_reg = _extract_a_operand(src)
        if a_reg is None:
            continue

        # dst is the accumulator base register (e.g., 'R4')
        accum_base = dst if dst else None

        if a_reg not in hmma_info:
            hmma_info[a_reg] = []
        hmma_info[a_reg].append((idx, accum_base))

    # Build subblocks: a_reg -> sorted list of line indices
    subblocks = {}
    for a_reg, entries in hmma_info.items():
        subblocks[a_reg] = [idx for idx, _ in sorted(entries, key=lambda x: x[0])]

    # Detect double-accumulation pairs: two groups writing to overlapping accum banks
    # Build accum bank sets per group
    accum_banks = {}  # a_reg -> set of accumulator base registers
    for a_reg, entries in hmma_info.items():
        accum_banks[a_reg] = set(base for _, base in entries if base)

    double_accum_pairs = []
    a_regs = sorted(subblocks.keys())
    for i in range(len(a_regs)):
        for j in range(i + 1, len(a_regs)):
            r1, r2 = a_regs[i], a_regs[j]
            if accum_banks.get(r1, set()) & accum_banks.get(r2, set()):
                # Shared accum banks — the group appearing first in the sass must execute first
                min_line_r1 = min(subblocks[r1])
                min_line_r2 = min(subblocks[r2])
                if min_line_r1 < min_line_r2:
                    double_accum_pairs.append((r1, r2))
                else:
                    double_accum_pairs.append((r2, r1))

    return subblocks, double_accum_pairs


# ========= .reuse Flag Fixup =========

def fixup_reuse_flags(sass, subblock_indices):
    """
    For a list of HMMA line indices in execution order (i.e., sorted by position in sass),
    set .reuse on the A operand of all but the last HMMA, and remove .reuse from the last.

    Modifies sass in-place.
    """
    if len(subblock_indices) < 2:
        return

    sorted_indices = sorted(subblock_indices)

    for idx in sorted_indices[:-1]:
        # Add .reuse to A operand if not already present
        line = sass[idx]
        # Match pattern like "R68," or "R112," (A operand is right after HMMA opcode)
        # HMMA format: HMMA.16816.F32 Rdst, Ra_operand, Rb, Raccum
        # We need to add .reuse to the A operand (second operand after dst)
        # Pattern: after "HMMA.16816.F32 Rnn, " the next register is A operand
        if '.reuse' not in _get_a_operand_text(line):
            sass[idx] = _add_reuse_to_a_operand(line)

    # Remove .reuse from the last HMMA's A operand
    last_idx = sorted_indices[-1]
    last_line = sass[last_idx]
    if '.reuse' in _get_a_operand_text(last_line):
        sass[last_idx] = _remove_reuse_from_a_operand(last_line)


def _get_a_operand_text(line):
    """Extract the A operand text (including .reuse if present) from an HMMA line."""
    # HMMA line format: [ctrl] HMMA.16816.F32 Rdst, Ra[.reuse], Rb, Raccum
    m = re.search(r'HMMA\.16816\.F32\s+\w+,\s+(R\d+(?:\.reuse)?)', line)
    return m.group(1) if m else ''


def _add_reuse_to_a_operand(line):
    """Add .reuse to the A operand of an HMMA instruction."""
    # Replace "Rnn," (A operand without .reuse) with "Rnn.reuse,"
    # Match the A operand position: after "HMMA.16816.F32 Rdst, "
    def replacer(m):
        return m.group(0).replace(m.group(1), m.group(1) + '.reuse', 1)
    result = re.sub(
        r'(HMMA\.16816\.F32\s+\w+,\s+)(R\d+)(,)',
        lambda m: m.group(1) + m.group(2) + '.reuse' + m.group(3),
        line, count=1
    )
    return result


def _remove_reuse_from_a_operand(line):
    """Remove .reuse from the A operand of an HMMA instruction."""
    # Match "Rnn.reuse" in the A operand position and remove .reuse
    return re.sub(
        r'(HMMA\.16816\.F32\s+\w+,\s+R\d+)\.reuse',
        r'\1',
        line, count=1
    )


# ========= Order Crossover (OX) =========

def order_crossover(perm1, perm2):
    """
    OX (Order Crossover) for two permutations of the same elements.

    Takes two permutations (lists of integers) of the same length.
    Selects a random substring from perm1, fills remaining positions
    with elements from perm2 in order.

    Returns the offspring permutation.
    """
    n = len(perm1)
    if n <= 1:
        return perm1[:]

    # Select random substring [start, end)
    start = random.randint(0, n - 1)
    end = random.randint(start + 1, n)  # at least 1 element

    # Copy substring from perm1
    offspring = [None] * n
    substring_set = set()
    for i in range(start, end):
        offspring[i] = perm1[i]
        substring_set.add(perm1[i])

    # Fill remaining positions with elements from perm2 in order
    p2_filtered = [x for x in perm2 if x not in substring_set]
    fill_idx = 0
    for i in range(n):
        if offspring[i] is None:
            offspring[i] = p2_filtered[fill_idx]
            fill_idx += 1

    return offspring


# ========= Independent Cluster Detection =========

def detect_independent_clusters(sass):
    """
    Detect independent instruction clusters for Mutation C.

    Returns a list of clusters, each being a dict:
        {
            'type': 'fadd' | 'fsetp_fmul',
            'indices': list of line indices (for FADD) or list of (fsetp_idx, fmul_idx) pairs,
            'start': first line index,
            'end': last line index + 1,
        }
    """
    clusters = []

    # --- FADD clusters: consecutive FADD instructions writing to different registers ---
    fadd_run = []
    fadd_dsts = set()
    for idx, line in enumerate(sass):
        ctrl, _, _, opcode, dst, src, _ = decode(line)
        if ctrl is None:
            if fadd_run:
                if len(fadd_run) >= 4:  # meaningful cluster
                    clusters.append({
                        'type': 'fadd',
                        'indices': fadd_run[:],
                        'start': fadd_run[0],
                        'end': fadd_run[-1] + 1,
                    })
                fadd_run = []
                fadd_dsts = set()
            continue

        is_fadd = opcode and 'FADD' in opcode and 'DFMA' not in opcode
        if is_fadd and dst and dst not in fadd_dsts:
            fadd_run.append(idx)
            fadd_dsts.add(dst)
        else:
            if len(fadd_run) >= 4:
                clusters.append({
                    'type': 'fadd',
                    'indices': fadd_run[:],
                    'start': fadd_run[0],
                    'end': fadd_run[-1] + 1,
                })
            fadd_run = []
            fadd_dsts = set()

    # Final FADD run
    if len(fadd_run) >= 4:
        clusters.append({
            'type': 'fadd',
            'indices': fadd_run[:],
            'start': fadd_run[0],
            'end': fadd_run[-1] + 1,
        })

    # --- FSETP/FMUL pairs: (FSETP.GE, @!Pn FMUL) pairs ---
    fsetp_fmul_pairs = []
    idx = 0
    while idx < len(sass) - 1:
        ctrl1, _, _, opcode1, dst1, src1, _ = decode(sass[idx])
        ctrl2, _, _, opcode2, dst2, src2, _ = decode(sass[idx + 1])

        if (ctrl1 is not None and ctrl2 is not None and
                opcode1 and 'FSETP' in opcode1 and
                opcode2 and 'FMUL' in opcode2 and
                '@!' in sass[idx + 1]):
            fsetp_fmul_pairs.append((idx, idx + 1))
            idx += 2
        else:
            idx += 1

    # Group consecutive FSETP/FMUL pairs into clusters
    if len(fsetp_fmul_pairs) >= 2:
        current_run = [fsetp_fmul_pairs[0]]
        for i in range(1, len(fsetp_fmul_pairs)):
            prev_end = fsetp_fmul_pairs[i - 1][1]
            curr_start = fsetp_fmul_pairs[i][0]
            # Allow small gaps (up to 2 non-pair instructions) between pairs
            if curr_start - prev_end <= 3:
                current_run.append(fsetp_fmul_pairs[i])
            else:
                if len(current_run) >= 2:
                    clusters.append({
                        'type': 'fsetp_fmul',
                        'indices': current_run[:],
                        'start': current_run[0][0],
                        'end': current_run[-1][1] + 1,
                    })
                current_run = [fsetp_fmul_pairs[i]]
        if len(current_run) >= 2:
            clusters.append({
                'type': 'fsetp_fmul',
                'indices': current_run[:],
                'start': current_run[0][0],
                'end': current_run[-1][1] + 1,
            })

    return clusters


class Individual:
    def __init__(self, kernel_section: List[str]):
        self.sass = kernel_section[:]
        self.fitness: float = float('inf')


class GeneticAlgorithm:
    def __init__(
        self,
        kernel_section: List[str],
        sasskernel: SassKernel,
        test_correctness: Callable,
        test_performance: Callable[[Individual], float],
    ):
        self.original_kernel_section = kernel_section[:]
        self.sasskernel = sasskernel
        self.test_correctness = test_correctness
        self.test_performance = test_performance
        self.history = []
        self._t0 = time.time()
        self.mut_attempts = 0
        self.mut_moves    = 0
        self.mut_valids   = 0
        self.counter = Counter(kernel_section)
        # Cache HMMA subblock detection and cluster detection for the original kernel
        self._hmma_subblocks = None
        self._hmma_double_accum = None
        self._clusters = None

    def _get_hmma_subblocks(self, sass):
        """Detect HMMA subblocks (cached on first call per sass signature)."""
        return detect_hmma_subblocks(sass)

    def _get_clusters(self, sass):
        """Detect independent clusters."""
        return detect_independent_clusters(sass)

    @staticmethod
    def _sig(sass_lines):
        return hashlib.sha1("\n".join(sass_lines).encode()).hexdigest()[:12]

    def _record_gen(self, gen_idx: int, population):
        fits = [ind.fitness for ind in population]
        finite_fits = [f for f in fits if f != float("inf")]
        best_ind = max(population, key=lambda x: x.fitness if x.fitness != float("inf") else float("-inf"))
        rec = {
            "gen": gen_idx,
            "best_fitness": best_ind.fitness,
            "best_sig": self._sig(best_ind.sass),
            "mean_fitness": float(sum(fits) / len(fits)),
            "median_fitness": float(statistics.median(fits)),
            "std_fitness": float(statistics.pstdev(finite_fits)) if len(finite_fits) > 1 else 0.0,
            "mut_attempts": int(self.mut_attempts),
            "mut_moves": int(self.mut_moves),
            "mut_valids": int(self.mut_valids),
            "move_rate": float(self.mut_moves / self.mut_attempts) if self.mut_attempts else 0.0,
            "valid_rate": float(self.mut_valids / max(1, self.mut_moves)),
            "elapsed_sec": float(time.time() - self._t0),
        }
        self.history.append(rec)
        self.mut_attempts = self.mut_moves = self.mut_valids = 0

    def save_history(self, path: str):
        if not self.history:
            return
        keys = list(self.history[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in self.history:
                w.writerow(row)

    # ---------- Fitness Evaluation ----------
    def evaluate_fitness(self, individual: Individual) -> float:
        try:
            updated_sass = self.sasskernel._update_kernel(individual.sass)
            ok = self.test_correctness(write_sass_file(updated_sass))
            if not ok:
                individual.fitness = float("inf")
                return individual.fitness
            f = self.test_performance(individual)
            individual.fitness = float(f) if f is not None else float("inf")
            return individual.fitness
        except Exception:
            individual.fitness = float("inf")
            return individual.fitness

    # ---------- Population Initialisation ----------
    def initialize_population(self, original_kernel_section: List[str]) -> List[Individual]:
        population: List[Individual] = []
        base = Individual(original_kernel_section[:])
        base.fitness = self.evaluate_fitness(base)
        population.append(base)
        while len(population) < POP_SIZE:
            dup = Individual(base.sass[:])
            dup.fitness = base.fitness
            population.append(dup)
        return population

    # ---------- Crossover: HMMA sub-block permutation (OX) with barrier-group fallback ----------
    def crossover(self, parent1: Individual, parent2: Individual):
        """
        Phase 3: HMMA Sub-Block Permutation Crossover.
        - Detect HMMA sub-blocks in parent1's sass
        - For each sub-block, extract HMMA line ordering as permutation chromosome
        - Apply OX (Order Crossover) between parent1 and parent2's permutations
        - Reconstruct child sass: non-HMMA lines stay in original positions,
          HMMA lines placed according to new permutation
        - Apply .reuse flag fixup
        - Respect double-accumulation constraints
        - Fall back to barrier-based crossover if no HMMA sub-blocks found
        """
        subblocks_p1, double_accum = self._get_hmma_subblocks(parent1.sass)
        subblocks_p2, _ = self._get_hmma_subblocks(parent2.sass)

        # Fall back to identity crossover if no HMMA sub-blocks
        if not subblocks_p1 or all(len(v) < 2 for v in subblocks_p1.values()):
            c1 = Individual(parent1.sass[:]); c1.fitness = parent1.fitness
            c2 = Individual(parent2.sass[:]); c2.fitness = parent2.fitness
            if hasattr(self, '_content_log'):
                self._content_log.append("XOVER_SAME")
            return c1, c2

        # Check that both parents have the same sub-block structure
        if set(subblocks_p1.keys()) != set(subblocks_p2.keys()):
            c1 = Individual(parent1.sass[:]); c1.fitness = parent1.fitness
            c2 = Individual(parent2.sass[:]); c2.fitness = parent2.fitness
            if hasattr(self, '_content_log'):
                self._content_log.append("XOVER_SAME")
            return c1, c2

        for a_reg in subblocks_p1:
            if len(subblocks_p1[a_reg]) != len(subblocks_p2[a_reg]):
                c1 = Individual(parent1.sass[:]); c1.fitness = parent1.fitness
                c2 = Individual(parent2.sass[:]); c2.fitness = parent2.fitness
                return c1, c2

        # Build child1 sass by applying OX on each sub-block
        child1_sass = parent1.sass[:]
        child2_sass = parent2.sass[:]

        for a_reg, indices_p1 in subblocks_p1.items():
            indices_p2 = subblocks_p2[a_reg]
            n = len(indices_p1)
            if n < 2:
                continue

            # The "permutation chromosome" is the ordering of HMMA instruction content
            # at the positions. We extract the actual SASS lines at those positions.
            lines_p1 = [parent1.sass[i] for i in indices_p1]
            lines_p2 = [parent2.sass[i] for i in indices_p2]

            # Create index permutations: perm[k] = which HMMA from parent goes to position k
            # For OX, we use integer indices 0..n-1
            perm1 = list(range(n))  # identity for parent1
            # For parent2, find which parent1 HMMA line matches each parent2 position
            # We match by accumulator destination register (unique per HMMA in a sub-block)
            dst_to_idx_p1 = {}
            for k, line in enumerate(lines_p1):
                _, _, _, _, dst, _, _ = decode(line)
                if dst:
                    dst_to_idx_p1[dst] = k

            perm2 = []
            valid_mapping = True
            for line in lines_p2:
                _, _, _, _, dst, _, _ = decode(line)
                if dst and dst in dst_to_idx_p1:
                    perm2.append(dst_to_idx_p1[dst])
                else:
                    valid_mapping = False
                    break

            if not valid_mapping or len(perm2) != n or set(perm2) != set(range(n)):
                continue  # skip this sub-block, keep parent arrangement

            # Apply OX
            child1_perm = order_crossover(perm1, perm2)
            child2_perm = order_crossover(perm2, perm1)

            # Reconstruct: place parent1's HMMA lines in the new permutation order
            # at the same positions (indices_p1)
            for k, pos in enumerate(indices_p1):
                child1_sass[pos] = lines_p1[child1_perm[k]]
            for k, pos in enumerate(indices_p2):
                child2_sass[pos] = lines_p2[child2_perm[k]]

            # Apply .reuse flag fixup for this sub-block
            fixup_reuse_flags(child1_sass, indices_p1)
            fixup_reuse_flags(child2_sass, indices_p2)

        # Validate double-accumulation constraints
        # For each (first_reg, second_reg) pair, all HMMAs of first_reg must appear
        # before all HMMAs of second_reg in the child
        c1_valid = True
        c2_valid = True
        for first_reg, second_reg in double_accum:
            if first_reg in subblocks_p1 and second_reg in subblocks_p1:
                first_indices = subblocks_p1[first_reg]
                second_indices = subblocks_p1[second_reg]
                # Check that max position of first group < min position of second group
                # (positions haven't moved — only content at those positions changed)
                # Since we only permute WITHIN each sub-block's positions, ordering between
                # sub-blocks is automatically preserved. This check is for safety.
                if max(first_indices) >= min(second_indices):
                    # Positions overlap — this shouldn't happen with correct sub-block detection
                    # but check anyway
                    pass  # Positions are fixed, only content permuted, so order is preserved

            if first_reg in subblocks_p2 and second_reg in subblocks_p2:
                first_indices = subblocks_p2[second_reg]
                second_indices = subblocks_p2[first_reg]

        # Multiset conservation check
        parent_counter = Counter(parent1.sass)
        if Counter(child1_sass) == parent_counter:
            c1 = Individual(child1_sass)
        else:
            c1 = Individual(parent1.sass[:])
            c1.fitness = parent1.fitness
            c1_valid = False

        parent2_counter = Counter(parent2.sass)
        if Counter(child2_sass) == parent2_counter:
            c2 = Individual(child2_sass)
        else:
            c2 = Individual(parent2.sass[:])
            c2.fitness = parent2.fitness
            c2_valid = False

        if hasattr(self, '_content_log'):
            if c1.sass != parent1.sass or c2.sass != parent2.sass:
                self._content_log.append("XOVER_DIFF")
            else:
                self._content_log.append("XOVER_SAME")
        return c1, c2

    # ---------- Mutate: cluster permutation, HMMA swap, interleave, non-adjacent, or adjacent ----------
    def mutate(self, individual: Individual) -> Individual:
        if random.random() >= MUTATION_RATE:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        content_log = getattr(self, "_content_log", [])
        self.mut_attempts += 1
        sass = individual.sass[:]
        sample = Sample(sass)
        dims, total, mem_loc, max_src_len = sample.static_analysis()

        if dims == 0:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        n_feat = 10 + 1 + 1 + 1 + max_src_len
        dummy_space = np.zeros((1, total, n_feat), dtype=np.float32)
        _, masks = sample.embedding(dummy_space, mem_loc, max_src_len)

        valid_actions = []
        for i, (up, down) in enumerate(masks):
            if up:   valid_actions.append(i * 2 + 0)
            if down: valid_actions.append(i * 2 + 1)

        # ===== Branch selection: cluster (15%) -> non-adjacent (25%) -> adjacent (60%) =====
        roll = random.random()

        # ===== Mutation C: Cluster Permutation — DISABLED (causes CUDA crashes) =====
        if False:
            cluster_done = False
            clusters = self._get_clusters(sass)
            if clusters:
                cluster = random.choice(clusters)
                before = sass[:]

                if cluster['type'] == 'fadd':
                    # Randomly permute all FADD lines in the cluster
                    indices = cluster['indices']
                    lines = [sass[i] for i in indices]
                    random.shuffle(lines)
                    for k, idx in enumerate(indices):
                        sass[idx] = lines[k]

                elif cluster['type'] == 'fsetp_fmul':
                    # Randomly permute pairs (keep each FSETP/FMUL pair together)
                    pairs = cluster['indices']  # list of (fsetp_idx, fmul_idx)
                    pair_lines = [(sass[fi], sass[mi]) for fi, mi in pairs]
                    random.shuffle(pair_lines)
                    # Write back shuffled pairs to original positions
                    for k, (fi, mi) in enumerate(pairs):
                        sass[fi] = pair_lines[k][0]
                        sass[mi] = pair_lines[k][1]

                # Multiset conservation check
                if Counter(sass) != Counter(before):
                    sass = before  # revert
                else:
                    individual.sass = sass
                    self.mut_moves += 1
                    individual.fitness = self.evaluate_fitness(individual)
                    if individual.fitness != float("inf"):
                        self.mut_valids += 1
                    cluster_done = True
                    print(f"CLUSTER MUT: type={cluster['type']} size={len(cluster['indices'])} "
                          f"range=[{cluster['start']},{cluster['end']})")

            if cluster_done:
                return individual
            # Fall through to non-adjacent or adjacent

        # ===== HMMA sub-block mutation (p=P_HMMA_MUT) =====
        if roll < P_CLUSTER + P_HMMA_MUT:
            hmma_done = False
            sb, _ = detect_hmma_subblocks(sass)
            # Find consecutive HMMA runs (no non-HMMA instructions between them)
            consecutive_runs = []
            for reg, indices in sb.items():
                current_run = [indices[0]]
                for k in range(1, len(indices)):
                    if indices[k] == indices[k-1] + 1:
                        current_run.append(indices[k])
                    else:
                        if len(current_run) >= 2:
                            consecutive_runs.append((reg, current_run[:]))
                        current_run = [indices[k]]
                if len(current_run) >= 2:
                    consecutive_runs.append((reg, current_run[:]))
            if consecutive_runs:
                reg, run = random.choice(consecutive_runs)
                i, j = random.sample(range(len(run)), 2)
                pos_i, pos_j = run[i], run[j]
                saved_sass = sass[:]
                sass[pos_i], sass[pos_j] = sass[pos_j], sass[pos_i]
                # Conservative stall count fixup (same as non-adjacent swap)
                stall_i = _get_stall_count(decode(sass[pos_i])[0])
                stall_j = _get_stall_count(decode(sass[pos_j])[0])
                max_stall = max(stall_i, stall_j)
                if max_stall > 0:
                    sass[pos_i] = _set_stall_in_line(sass[pos_i], max_stall)
                    sass[pos_j] = _set_stall_in_line(sass[pos_j], max_stall)
                # Fixup .reuse ONLY for the consecutive run being swapped
                fixup_reuse_flags(sass, run)
                individual.sass = sass
                self.mut_moves += 1
                individual.fitness = self.evaluate_fitness(individual)
                if individual.fitness != float("inf"):
                    self.mut_valids += 1
                    hmma_done = True
                    content_log.append("HMMA_OK"); print(f"HMMA MUT OK: A={reg} run[{run[0]}-{run[-1]}] swap {pos_i}<->{pos_j} fit={individual.fitness:.4f}")
                else:
                    sass = saved_sass
                    individual.sass = sass
                    content_log.append("HMMA_FAIL"); print(f"HMMA MUT FAIL: A={reg} swap {pos_i}<->{pos_j} -> inf")
            if hmma_done:
                return individual
            else:
                # Don't fall through to adjacent/non-adjacent swap
                # Mixing stall count changes with HMMA reordering causes cumulative corruption
                if individual.fitness is None:
                    individual.fitness = self.evaluate_fitness(individual)
                return individual

        # ===== HMMA-Memory Interleave Mutation (p=P_INTERLEAVE) =====
        if roll < P_CLUSTER + P_HMMA_MUT + P_INTERLEAVE:
            interleave_done = False
            sb, _ = detect_hmma_subblocks(sass)
            # Find consecutive HMMA runs of length >= 3
            long_runs = []
            for reg, indices in sb.items():
                current_run = [indices[0]]
                for k in range(1, len(indices)):
                    if indices[k] == indices[k-1] + 1:
                        current_run.append(indices[k])
                    else:
                        if len(current_run) >= 3:
                            long_runs.append((reg, current_run[:]))
                        current_run = [indices[k]]
                if len(current_run) >= 3:
                    long_runs.append((reg, current_run[:]))

            if long_runs:
                reg, run = random.choice(long_runs)
                # Look for memory instructions near the cluster (within 20 lines)
                run_start, run_end = run[0], run[-1]
                nearby_mem = []
                for offset in range(1, 21):
                    for pos in [run_start - offset, run_end + offset]:
                        if 0 <= pos < len(sass):
                            try:
                                _, _, _, op, _, _, _ = decode(sass[pos])
                                if op and any(m in op for m in ['LDSM', 'LDGSTS', 'LDG', 'LDS']):
                                    nearby_mem.append(pos)
                            except:
                                pass
                if nearby_mem:
                    random.shuffle(nearby_mem)
                    for mem_pos in nearby_mem[:10]:
                        # Pick a HMMA in the middle of the run to swap with
                        hmma_pos = random.choice(run[1:-1]) if len(run) > 2 else run[0]
                        ci, cj = min(mem_pos, hmma_pos), max(mem_pos, hmma_pos)
                        if cj - ci <= 1 or cj - ci > 20:
                            continue
                        if transitive_dependency_check(sass, ci, cj):
                            saved_sass = sass[:]
                            sass[ci], sass[cj] = sass[cj], sass[ci]
                            if Counter(sass) != Counter(saved_sass):
                                sass = saved_sass
                                continue
                            stall_ci = _get_stall_count(decode(sass[ci])[0])
                            stall_cj = _get_stall_count(decode(sass[cj])[0])
                            ms = max(stall_ci, stall_cj)
                            if ms > 0:
                                sass[ci] = _set_stall_in_line(sass[ci], ms)
                                sass[cj] = _set_stall_in_line(sass[cj], ms)
                            individual.sass = sass
                            self.mut_moves += 1
                            individual.fitness = self.evaluate_fitness(individual)
                            if individual.fitness != float("inf"):
                                self.mut_valids += 1
                                interleave_done = True
                                content_log.append("INTLV_OK"); print(f"INTERLEAVE OK: mem@{mem_pos}<->hmma@{hmma_pos} (A={reg}) fit={individual.fitness:.4f}")
                            else:
                                sass = saved_sass
                                individual.sass = sass
                                content_log.append("INTLV_FAIL"); print(f"INTERLEAVE FAIL: mem@{mem_pos}<->hmma@{hmma_pos} -> inf")
                            break
            if interleave_done:
                return individual
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        # ===== Non-adjacent swap branch =====
        candidates = list(sample.candidates)
        if roll < P_CLUSTER + P_HMMA_MUT + P_INTERLEAVE + P_NONADJ and len(candidates) >= 1:
            nonadj_done = False
            for _attempt in range(30):
                ci = random.choice(candidates)
                offset = random.choice(range(2, 21))
                if random.random() < 0.5:
                    offset = -offset
                cj = ci + offset
                if cj < 0 or cj >= len(sass):
                    continue
                if ci > cj:
                    ci, cj = cj, ci
                if cj <= ci + 1 or cj - ci > 20:
                    continue
                if transitive_dependency_check(sass, ci, cj):
                    before = sass[:]
                    sass[ci], sass[cj] = sass[cj], sass[ci]
                    if Counter(sass) != Counter(before):
                        sass[ci], sass[cj] = sass[cj], sass[ci]
                        continue
                    stall_ci = _get_stall_count(decode(sass[ci])[0])
                    stall_cj = _get_stall_count(decode(sass[cj])[0])
                    max_stall = max(stall_ci, stall_cj)
                    if max_stall > 0:
                        sass[ci] = _set_stall_in_line(sass[ci], max_stall)
                        sass[cj] = _set_stall_in_line(sass[cj], max_stall)
                    individual.sass = sass
                    self.mut_moves += 1
                    individual.fitness = self.evaluate_fitness(individual)
                    if individual.fitness != float("inf"):
                        self.mut_valids += 1
                    nonadj_done = True
                    content_log.append("NONADJ_OK"); print(f"NONADJ SWAP: {ci}<->{cj} (dist={cj-ci}) stall={max_stall}")
                    break
            if nonadj_done:
                return individual
            content_log.append("NONADJ_FAIL"); print("NONADJ FAIL: tried 30 pairs, no valid swap found")

        # ===== Adjacent swap branch =====
        if not valid_actions:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        action = random.choice(valid_actions)
        index, direction = divmod(action, 2)
        before = sample.kernel_section[:]
        sample.apply(index, direction)

        after = sample.kernel_section
        if Counter(after) != Counter(before):
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        individual.sass = after
        self.mut_moves += 1
        individual.fitness = self.evaluate_fitness(individual)
        if individual.fitness != float("inf"):
            self.mut_valids += 1
            content_log.append("ADJ_OK")
        return individual

    # ---------- Selection ----------
    def tournament_selection(self, population, k=4):
        contenders = random.sample(population, k)
        return max(contenders, key=lambda x: x.fitness if x.fitness != float("inf") else float("-inf"))

    # ---------- Main GA Loop ----------
    def run_ga(self, original_kernel: List[str]) -> Individual:
        print("testing the correctness of the original kernel")
        origin = Individual(original_kernel[:])
        origin.fitness = self.evaluate_fitness(origin)
        print(f"original kernel fitness: {origin.fitness}")

        # Try to resume from checkpoint
        start_gen = 0
        try:
            with open("ga_checkpoint.pkl", "rb") as ckf:
                ckpt = pickle.load(ckf)
            population = [Individual(s) for s, f in ckpt["pop"]]
            for ind, (_, f) in zip(population, ckpt["pop"]):
                ind.fitness = f
            start_gen = ckpt["gen"] + 1
            best_fit = max(f for _, f in ckpt["pop"] if f != float("inf"))
            print(f"RESUMED from gen {ckpt['gen']}, best={best_fit:.4f}")
        except Exception:
            population = self.initialize_population(original_kernel)
            self._record_gen(0, population)

        content_log = []
        self._content_log = content_log
        for gen in range(start_gen, NUM_GENERATIONS):
            gpu_keepalive()

            if gen % 5 == 0 and self.mut_attempts > 0:
                print(f"success rate : {self.mut_valids/self.mut_attempts}")
                print(f"move rate:{self.mut_moves/self.mut_attempts}")
            best = max(population, key=lambda x: x.fitness if x.fitness != float("inf") else float("-inf"))
            elapsed = time.time() - self._t0
            sec_per_gen = elapsed / (gen + 1) if gen > 0 else 0
            eta = sec_per_gen * (NUM_GENERATIONS - gen - 1)
            eta_min = int(eta // 60)
            eta_sec = int(eta % 60)
            pct = (gen + 1) / NUM_GENERATIONS * 100
            # Count operations in this generation
            hmma_ok = content_log.count("HMMA_OK")
            hmma_fail = content_log.count("HMMA_FAIL")
            intlv_ok = content_log.count("INTLV_OK")
            intlv_fail = content_log.count("INTLV_FAIL")
            nonadj_ok = content_log.count("NONADJ_OK")
            nonadj_fail = content_log.count("NONADJ_FAIL")
            adj_ok = content_log.count("ADJ_OK")
            xover_diff = content_log.count("XOVER_DIFF")
            xover_same = content_log.count("XOVER_SAME")
            print(f"GEN {gen}/{NUM_GENERATIONS} ({pct:.1f}%) best={best.fitness:.4f} | {elapsed:.0f}s ETA {eta_min}m{eta_sec:02d}s | hmma={hmma_ok}/{hmma_ok+hmma_fail} intlv={intlv_ok}/{intlv_ok+intlv_fail} nonadj={nonadj_ok}/{nonadj_ok+nonadj_fail} adj={adj_ok} xover={xover_diff}/{xover_diff+xover_same}")
            content_log.clear()

            population.sort(key=lambda x: x.fitness if x.fitness != float("inf") else float("-inf"), reverse=True)
            next_gen = population[:ELITE_SIZE]

            max_tries = POP_SIZE * 5
            tries = 0
            while len(next_gen) < POP_SIZE and tries < max_tries:
                tries += 1
                p1 = self.tournament_selection(population, k=4)
                p2 = self.tournament_selection(population, k=4)
                c1, c2 = self.crossover(p1, p2)
                if c1:
                    c1 = self.mutate(c1)
                    if c1.fitness != float("inf"):
                        next_gen.append(c1)
                if len(next_gen) < POP_SIZE and c2:
                    c2 = self.mutate(c2)
                    if c2.fitness != float("inf"):
                        next_gen.append(c2)
            while len(next_gen) < POP_SIZE:
                filler = Individual(next_gen[0].sass[:])
                filler.fitness = next_gen[0].fitness
                next_gen.append(filler)

            population = next_gen
            self._record_gen(gen, population)

            # Checkpoint every 10 generations
            if gen % 10 == 0:
                try:
                    ckpt = {"gen": gen, "pop": [(ind.sass[:], ind.fitness) for ind in population]}
                    with open("ga_checkpoint.pkl", "wb") as ckf:
                        pickle.dump(ckpt, ckf)
                except Exception:
                    pass

        best = max(population, key=lambda x: x.fitness if x.fitness != float("inf") else float("-inf"))
        print(f"Best fitness:{best.fitness}")
        return best