
"""
Matrix-vector multiplication x = M.v using MapReduce
(Hadoop Streaming style).

Input records: i<TAB>j<TAB>m_ij

Hadoop:
-mapper "python exp15.py map"
-reducer "python exp15.py reduce"
-files exp15.py,vector.txt
"""

import sys
from itertools import groupby
import numpy as np


# ---------------- Load Vector ----------------

def load_vector(path="vector.txt"):
    with open(path) as f:
        return {
            int(j): float(v)
            for j, v in (line.split() for line in f)
        }


# ---------------- Mapper Function ----------------

def mapper(lines, v):
    for line in lines:
        i, j, m = line.split()
        yield f"{int(i)}\t{float(m) * v[int(j)]}"


# ---------------- Reducer Function ----------------

def reducer(lines):
    pairs = (
        l.rstrip("\n").split("\t")
        for l in lines
    )

    for i, grp in groupby(
        pairs,
        key=lambda kv: int(kv[0])
    ):
        yield f"{i}\t{sum(float(x) for _, x in grp)}"


# ---------------- Hadoop Streaming Mode ----------------

if len(sys.argv) > 1 and sys.argv[1] in ("map", "reduce"):

    if sys.argv[1] == "map":
        out = mapper(sys.stdin, load_vector())
    else:
        out = reducer(sys.stdin)

    for o in out:
        print(o)

    sys.exit()


# ---------------- Local Simulation ----------------

# Link matrix of a 5-page website
# Column j = page j shares its rank equally
# among the pages it links to.

M = np.array([
    [0,     0.5, 0,   0,   1/3],
    [1/3,   0,   0,   0.5, 0],
    [1/3,   0,   0,   0.5, 1/3],
    [1/3,   0,   1.0, 0,   1/3],
    [0,     0.5, 0,   0,   0]
])

v = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Current PageRank estimate


# Convert non-zero matrix entries into records

entries = [
    f"{i}\t{j}\t{M[i, j]}"
    for i in range(5)
    for j in range(5)
    if M[i, j] != 0
]


# Save matrix and vector into files

with open("matrix.txt", "w") as f:
    f.write("\n".join(entries))

with open("vector.txt", "w") as f:
    f.write(
        "\n".join(
            f"{j} {val}"
            for j, val in enumerate(v)
        )
    )

vec = load_vector()


# ---------------- Display Information ----------------

print("Matrix-Vector Multiplication using MapReduce (PageRank step)")
print("=" * 60)

print("Matrix M (5 x 5):")
print(np.round(M, 3))

print("Vector v:", v)

print(
    f"Non-zero entries stored as (i, j, m_ij): "
    f"{len(entries)} of {M.size}"
)


# ---------------- MAP PHASE ----------------

splits = [
    entries[:5],
    entries[5:]
]

print("\n--- MAP PHASE: emit (i, m_ij * v_j) ---")

mapped = []

for s, chunk in enumerate(splits, 1):
    out = list(mapper(chunk, vec))
    mapped += out

    print(f"Mapper {s}:")

    for rec, o in zip(chunk, out):
        i, j, m = rec.split("\t")

        print(
            f"  ({i},{j}, {float(m):.3f}) "
            f"-> key {o.split()[0]}, "
            f"value {float(o.split()[1]):.4f}"
        )


# ---------------- SHUFFLE AND SORT ----------------

print("\n--- SHUFFLE AND SORT: group values by row i ---")

shuffled = sorted(
    mapped,
    key=lambda s: int(s.split("\t")[0])
)

for i, grp in groupby(
    shuffled,
    key=lambda s: s.split("\t")[0]
):
    print(
        f"  row {i}: "
        f"{[round(float(g.split()[1]), 4) for g in grp]}"
    )


# ---------------- REDUCE PHASE ----------------

print("\n--- REDUCE PHASE: x_i = sum of values ---")

x = np.zeros(5)

for r in reducer(shuffled):
    i, val = r.split("\t")
    x[int(i)] = float(val)

    print(f"  x[{i}] = {float(val):.4f}")


# ---------------- VERIFY RESULT ----------------

print("\nMapReduce result x  :", np.round(x, 4))

print("NumPy check   M @ v :", np.round(M @ v, 4))

print("Results match       :", np.allclose(x, M @ v))


# ---------------- PAGERANK POWER ITERATION ----------------

print("\nRepeating the job (power iteration) gives PageRank:")

r = v.copy()

for it in range(1, 31):
    r_new = M @ r

    if it in (1, 2, 5, 30):
        print(f"  iteration {it:>2}: {np.round(r_new, 4)}")

    r = r_new

print("Most important page:", int(np.argmax(r)))
