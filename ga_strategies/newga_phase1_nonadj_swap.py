# newga_phase1_nonadj_swap.py — Adjacent + non-adjacent swap mutation (Report Section 4.1.1-4.1.2)
from typing import List, Optional, Callable
from collections import Counter
import random
import re
import time, csv, hashlib, statistics
import numpy as np
import pickle

from sass_kernel import SassKernel
from sassgen import write_sass_file
from decoder import decode, decode_ctrl_code
from gpu_utils import get_gpu_cc, get_mutatable_ops

from sample import Sample

# ========= Hyperparameters =========
POP_SIZE        = 10
MUTATION_RATE   = 1.0
NUM_GENERATIONS = 100
ELITE_SIZE      = 4
P_NONADJ        = 0.3
MULTI_MUT_STEPS = 5

_CC = get_gpu_cc()
_, _BAN_OPS = get_mutatable_ops(_CC)


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


# ========= Register Expansion =========

def _expand_registers(regs, opcode):
    """
    Expand base registers to include implicit adjacent registers.
    HMMA uses 4-reg groups (dst, A operand) or 2-reg groups (B operand).
    LDSM uses 4-reg groups.
    LDG/STG/LDGSTS with .128 suffix use 4-reg groups.
    For safety, expand all registers by +0,+1,+2,+3 for wide instructions,
    and +0,+1 for others.
    """
    expanded = set()
    is_wide = False
    if opcode:
        is_wide = any(x in opcode for x in ['HMMA', 'LDSM', '.128', 'LDGSTS'])

    for r in regs:
        if not r.startswith('R'):
            expanded.add(r)
            continue
        # Extract register number
        base = r.replace('.reuse', '')
        try:
            num = int(base[1:])
        except ValueError:
            expanded.add(r)
            continue
        if is_wide:
            for offset in range(4):
                expanded.add(f'R{num + offset}')
        else:
            # At minimum, add adjacent register (pairs)
            expanded.add(f'R{num}')
            expanded.add(f'R{num ^ 1}')  # XOR 1 gives adjacent pair
    return expanded


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
    writes_i_raw = set()
    if dst_i and (dst_i.startswith('R') or dst_i.startswith('P') or dst_i.startswith('UR')):
        writes_i_raw.add(dst_i)
    writes_j_raw = set()
    if dst_j and (dst_j.startswith('R') or dst_j.startswith('P') or dst_j.startswith('UR')):
        writes_j_raw.add(dst_j)
    reads_i_raw = set()
    for s in (src_i or []):
        if s.startswith('R') or s.startswith('P') or s.startswith('UR'):
            reads_i_raw.add(s)
    reads_j_raw = set()
    for s in (src_j or []):
        if s.startswith('R') or s.startswith('P') or s.startswith('UR'):
            reads_j_raw.add(s)
    # Expand to cover implicit adjacent registers
    writes_i = _expand_registers(writes_i_raw, opcode_i)
    writes_j = _expand_registers(writes_j_raw, opcode_j)
    reads_i = _expand_registers(reads_i_raw, opcode_i)
    reads_j = _expand_registers(reads_j_raw, opcode_j)
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
        writes_k_raw = set()
        if dst_k and (dst_k.startswith('R') or dst_k.startswith('P') or dst_k.startswith('UR')):
            writes_k_raw.add(dst_k)
        reads_k_raw = set()
        for s in (src_k or []):
            if s.startswith('R') or s.startswith('P') or s.startswith('UR'):
                reads_k_raw.add(s)
        writes_k = _expand_registers(writes_k_raw, opcode_k)
        reads_k = _expand_registers(reads_k_raw, opcode_k)
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

    # ---------- Crossover: identity (returns parent copies) ----------
    def crossover(self, parent1: Individual, parent2: Individual):
        c1 = Individual(parent1.sass[:]); c1.fitness = parent1.fitness
        c2 = Individual(parent2.sass[:]); c2.fitness = parent2.fitness
        return c1, c2

    # ---------- Mutate: safe swap (adjacent or non-adjacent) ----------
    def _single_swap(self, sass):
        """Perform one adjacent or non-adjacent swap. Returns modified sass or None if failed."""
        sample = Sample(sass)
        dims, total, mem_loc, max_src_len = sample.static_analysis()
        if dims == 0:
            return None

        n_feat = 10 + 1 + 1 + 1 + max_src_len
        dummy_space = np.zeros((1, total, n_feat), dtype=np.float32)
        _, masks = sample.embedding(dummy_space, mem_loc, max_src_len)

        valid_actions = []
        for i, (up, down) in enumerate(masks):
            if up:   valid_actions.append(i * 2 + 0)
            if down: valid_actions.append(i * 2 + 1)

        candidates = list(sample.candidates)
        if random.random() < P_NONADJ and len(candidates) >= 1:
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
                    return sass
            # fallback to adjacent

        if not valid_actions:
            return None

        action = random.choice(valid_actions)
        index, direction = divmod(action, 2)
        before = sample.kernel_section[:]
        sample.apply(index, direction)
        after = sample.kernel_section
        if Counter(after) != Counter(before):
            return None
        return after

    def mutate(self, individual: Individual) -> Individual:
        if random.random() >= MUTATION_RATE:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        self.mut_attempts += 1
        sass = individual.sass[:]

        # Apply multiple swap steps
        steps_done = 0
        for _ in range(MULTI_MUT_STEPS):
            result = self._single_swap(sass)
            if result is not None:
                sass = result
                steps_done += 1

        if steps_done == 0:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
            return individual

        individual.sass = sass
        self.mut_moves += 1
        individual.fitness = self.evaluate_fitness(individual)
        if individual.fitness != float("inf"):
            self.mut_valids += 1
            if steps_done > 1:
                print(f"MULTI-MUT: {steps_done} steps applied")
        return individual
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

        # ===== Non-adjacent swap branch (p=P_NONADJ) =====
        candidates = list(sample.candidates)
        if random.random() < P_NONADJ and len(candidates) >= 1:
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
                    print(f"NONADJ SWAP: {ci}<->{cj} (dist={cj-ci}) stall={max_stall}")
                    break
            if nonadj_done:
                return individual
            print("NONADJ FAIL: tried 30 pairs, no valid swap found")

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

        start_gen = 0
        try:
            with open("ga_checkpoint.pkl", "rb") as ckf:
                ckpt = pickle.load(ckf)
            population = [Individual(s) for s, f in ckpt["pop"]]
            for ind, (_, f) in zip(population, ckpt["pop"]):
                ind.fitness = f
            start_gen = ckpt["gen"] + 1
            print(f"RESUMED from gen {ckpt['gen']}, best={max(f for _, f in ckpt['pop'] if f != float('inf')):.4f}")
        except Exception:
            population = self.initialize_population(original_kernel)
            self._record_gen(0, population)
        for gen in range(start_gen, NUM_GENERATIONS):
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
            print(f"GEN {gen}/{NUM_GENERATIONS} ({pct:.1f}%) best={best.fitness:.4f} | {elapsed:.0f}s elapsed, ETA {eta_min}m{eta_sec:02d}s")

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
            # Fill remaining slots with copies of the best individual
            while len(next_gen) < POP_SIZE:
                filler = Individual(next_gen[0].sass[:])
                filler.fitness = next_gen[0].fitness
                next_gen.append(filler)

            population = next_gen
            self._record_gen(gen, population)
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