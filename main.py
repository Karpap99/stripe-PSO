import numpy as np
import math
from mealpy import FloatVar, PSO
N = 10
R = [0.1] * N
L = 0.4
H = 1

def objective_stripe(solution):
    x = np.reshape(solution, (-1, 2))
    right = np.zeros(N, dtype=float)
    shtraf = 0
    for i in range(N):
        right[i] = x[i][0] + R[i]
        if x[i][0] < R[i]:
            shtraf += abs(x[i][0] - R[i])
        if x[i][1] > (1 - R[i]):
            shtraf += abs(1 - x[i][1] - R[i])
        if x[i][1] < R[i]:
            shtraf += abs(x[i][1] - R[i])
        for j in range(i + 1, N):
            hyp = math.dist(x[i][:], x[j][:])
            if hyp < (R[i] + R[j]):
                shtraf += 2 * abs(hyp - R[i] - R[j])
    return max(right) + shtraf

problem_dict = {
    "bounds": FloatVar(lb=(0.,)*2*N, ub=(1.,)*2*N, name="delta"),
    "obj_func": objective_stripe,
    "minmax": "min"
}

model = PSO.CL_PSO(epoch=2000, pop_size=200, c_local=1.2, w_min=L, w_max=H, max_flag=7)
g_best = model.solve(problem_dict)

print(f"Solution:{g_best.solution}, fitness:{g_best.target.fitness}")