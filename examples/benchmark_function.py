"""Linear versus nonlinear fusion on an analytic two-level test function (Perdikaris et al., 2017).

    f_low(x)  = sin(8 pi x)
    f_high(x) = (x - sqrt(2)) * f_low(x)**2

50 low-fidelity and 14 high-fidelity points. The high-fidelity function depends nonlinearly on the
low-fidelity one, the situation that Eqs. (10)-(11) are built for.

    python examples/benchmark_function.py
"""
import numpy as np

import mfgpr_framework as m

f_low = lambda x: np.sin(8 * np.pi * x)
f_high = lambda x: (x - np.sqrt(2)) * f_low(x) ** 2

xl, xh = np.linspace(0, 1, 50), np.linspace(0, 1, 14)
levels = [(xl, f_low(xl)), (xh, f_high(xh))]
xs = np.linspace(0, 1, 500)
for kind in m.KINDS:
    model = m.make_model(kind, m.ModelOptions(restarts=5, n_samples=100)).fit(levels)
    mean, std = model.predict(xs, include_noise=False)
    rmse = np.sqrt(np.mean((mean - f_high(xs)) ** 2))
    print(f"{m.LABELS[kind]:<17} RMSE against f_high: {rmse:.4f}   mean predictive std: {std.mean():.4f}")
