import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


def compute_metrics(pred, exact, x_grid=None, t_grid=None):
    diff = pred - exact
    l2_num = np.sqrt(np.mean(diff**2))
    l2_den = np.sqrt(np.mean(exact**2))
    l2_rel = l2_num / l2_den if l2_den > 1e-12 else l2_num

    h1_rel = l2_rel
    if x_grid is not None and t_grid is not None and pred.ndim == 2:
        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
        dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else 1.0
        grad_x_pred  = np.gradient(pred,  dx, axis=1)
        grad_t_pred  = np.gradient(pred,  dt, axis=0)
        grad_x_exact = np.gradient(exact, dx, axis=1)
        grad_t_exact = np.gradient(exact, dt, axis=0)
        h1_num = np.sqrt(np.mean(diff**2) +
                         np.mean((grad_x_pred - grad_x_exact)**2) +
                         np.mean((grad_t_pred - grad_t_exact)**2))
        h1_den = np.sqrt(np.mean(exact**2) +
                         np.mean(grad_x_exact**2) +
                         np.mean(grad_t_exact**2))
        h1_rel = h1_num / h1_den if h1_den > 1e-12 else h1_num

    linf = np.max(np.abs(diff))
    mae  = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))


    return dict(l2_rel=l2_rel, h1_rel=h1_rel, linf=linf,
                mae=mae, rmse=rmse)


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix} Метрики качества:")
    print(f"  L2 relative error:    {metrics['l2_rel']*100:.4f}%")
    print(f"  H1 relative error:    {metrics['h1_rel']*100:.4f}%")
    print(f"  L∞ (max error):       {metrics['linf']:.6f}")
    print(f"  MAE:                  {metrics['mae']:.6f}")
    print(f"  RMSE:                 {metrics['rmse']:.6f}")


_PDE_REGISTRY = {
    "heat": {
        "operator": lambda u, u_x, u_xx, p: p["k"] * u_xx,
        "defaults": {"k": 0.3},
        "description": "u_t = {k}·u_xx + f",
    },
    "diffusion": {
        "operator": lambda u, u_x, u_xx, p: p["D"] * u_xx,
        "defaults": {"D": 0.1},
        "description": "u_t = {D}·u_xx + f",
    },
    "advection_diffusion": {
        "operator": lambda u, u_x, u_xx, p: p["k"] * u_xx - p["v"] * u_x,
        "defaults": {"k": 0.1, "v": 1.0},
        "description": "u_t = {k}·u_xx - {v}·u_x + f",
    },
    "reaction_diffusion": {
        "operator": lambda u, u_x, u_xx, p: p["k"] * u_xx + p["alpha"] * u * (1.0 - u),
        "defaults": {"k": 0.1, "alpha": 1.0},
        "description": "u_t = {k}·u_xx + {alpha}·u·(1-u) + f",
    },
}


def get_pde_config(pde_type, **kwargs):
    if pde_type not in _PDE_REGISTRY:
        raise ValueError(f"Неизвестный тип PDE: {pde_type}. Доступные: {', '.join(_PDE_REGISTRY)}")
    entry  = _PDE_REGISTRY[pde_type]
    params = {k: kwargs.get(k, v) for k, v in entry["defaults"].items()}
    return dict(type=pde_type, operator=entry["operator"], params=params,
                description=entry["description"].format(**params))


def make_consistent_config(
    u_exact,
    pde_type="heat",
    pde_kwargs=None,
    x_range=(0.0, 1.0),
    t_range=(0.0, 1.0),

    bc_left_type="dirichlet", 
    bc_right_type="dirichlet",

    n_collocation=5000,
    n_boundary=1000,
    n_initial=1000,
    n_observations=900,
    noise_level=0.001,

    hidden_layers_u=(64, 64, 64, 64),
    hidden_layers_f=(64, 64, 64),
    activation="tanh",
    learning_rate=5e-4,
    epochs=20000,
    print_every=2000,
    eval_every=1000,
):

    if pde_kwargs is None:
        pde_kwargs = {}

    for side, bc_type in [("left", bc_left_type), ("right", bc_right_type)]:
        if bc_type not in ("dirichlet", "neumann"):
            raise ValueError(f"bc_{side}_type должен быть 'dirichlet' или 'neumann', получено: '{bc_type}'")

    pde_cfg   = get_pde_config(pde_type, **pde_kwargs)
    pde_op    = pde_cfg["operator"]
    pde_params = pde_cfg["params"]
    eps = 1e-6

    def f_true(x, t):
        x = np.asarray(x, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        u_val = u_exact(x, t)
        u_t   = (u_exact(x, t + eps) - u_exact(x, t - eps)) / (2 * eps)
        u_x   = (u_exact(x + eps, t) - u_exact(x - eps, t)) / (2 * eps)
        u_xx  = (u_exact(x + eps, t) - 2 * u_val + u_exact(x - eps, t)) / eps**2
        return u_t - pde_op(u_val, u_x, u_xx, pde_params)

    def bc_left_val_fn(t):
        t = np.asarray(t, dtype=np.float64)
        x0 = np.zeros_like(t)
        if bc_left_type == "dirichlet":
            return u_exact(x0, t)
        else:
            return -(u_exact(x0 + eps, t) - u_exact(x0 - eps, t)) / (2 * eps)


    def bc_right_val_fn(t):
        t  = np.asarray(t, dtype=np.float64)
        x1 = np.ones_like(t) * x_range[1]
        if bc_right_type == "dirichlet":
            return u_exact(x1, t)
        else:
            return (u_exact(x1 + eps, t) - u_exact(x1 - eps, t)) / (2 * eps)

    return {
        "pde":   pde_cfg,
        "domain": {"x_range": list(x_range), "t_range": list(t_range)},
        "sampling": {
            "n_collocation":  n_collocation,
            "n_boundary":     n_boundary,
            "n_initial":      n_initial,
            "n_observations": n_observations,
            "noise_level":    noise_level,
        },
        "training": {
            "hidden_layers_u": list(hidden_layers_u),
            "hidden_layers_f": list(hidden_layers_f),
            "activation":      activation,
            "learning_rate":   learning_rate,
            "epochs":          epochs,
            "print_every":     print_every,
            "eval_every":      eval_every,
        },
        "u_exact":        u_exact,
        "f_true":         f_true,
        "bc_left_type":   bc_left_type,
        "bc_right_type":  bc_right_type,
        "bc_left_val_fn": bc_left_val_fn,
        "bc_right_val_fn": bc_right_val_fn,
    }


def make_activation(name):
    return {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}.get(name, nn.Tanh)()


def to_torch_fn(fn):
    def wrapper(x):
        x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
        dev  = x.device if isinstance(x, torch.Tensor) else device
        return torch.tensor(fn(x_np), dtype=torch.float32, device=dev)
    return wrapper


def prepare_config(cfg):
    c = cfg.copy()
    u  = cfg["u_exact"]
    c["initial_condition"]  = to_torch_fn(lambda x: u(x, np.zeros_like(x)))
    c["boundary_left_fn"]   = to_torch_fn(cfg["bc_left_val_fn"])
    c["boundary_right_fn"]  = to_torch_fn(cfg["bc_right_val_fn"])
    return c


def build_solution_grid(config, nx=101, nt=1001):
    xr = config["domain"]["x_range"]
    tr = config["domain"]["t_range"]
    x_grid = np.linspace(*xr, nx)
    t_grid = np.linspace(*tr, nt)
    X, T   = np.meshgrid(x_grid, t_grid)
    U      = config["u_exact"](X, T)
    return x_grid, t_grid, U


def generate_observations(x_grid, t_grid, U, config):
    smp   = config["sampling"]
    dom   = config["domain"]
    n_side = int(np.sqrt(smp["n_observations"]))
    x_sub = np.linspace(*dom["x_range"], n_side)
    t_sub = np.linspace(*dom["t_range"], n_side)
    Xm, Tm = np.meshgrid(x_sub, t_sub)
    xf, tf = Xm.flatten(), Tm.flatten()

    interp = RegularGridInterpolator((t_grid, x_grid), U)
    u_vals = interp(np.column_stack([tf, xf]))
    u_vals += smp["noise_level"] * np.random.randn(len(xf))

    return (
        torch.tensor(xf, dtype=torch.float32, device=device).unsqueeze(1),
        torch.tensor(tf, dtype=torch.float32, device=device).unsqueeze(1),
        torch.tensor(u_vals, dtype=torch.float32, device=device).unsqueeze(1),
    )



def generate_collocation_points(config):
    n   = config["sampling"]["n_collocation"]
    dom = config["domain"]
    x   = np.random.uniform(*dom["x_range"], n)
    t   = np.random.uniform(*dom["t_range"], n)
    return (
        torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True),
        torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(1).requires_grad_(True),
    )


def generate_boundary_points(config):
    smp = config["sampling"]
    dom = config["domain"]
    n_b = smp["n_boundary"]
    n_i = smp["n_initial"]

    t_bc = np.random.uniform(*dom["t_range"], n_b)
    t_left  = torch.tensor(t_bc, dtype=torch.float32, device=device).unsqueeze(1)
    t_right = torch.tensor(t_bc, dtype=torch.float32, device=device).unsqueeze(1)

    x_left  = torch.zeros(n_b, 1, device=device).requires_grad_(True)
    x_right = (torch.ones(n_b, 1, device=device) * dom["x_range"][1]).requires_grad_(True)

    g_left  = config["boundary_left_fn"](t_left)
    g_right = config["boundary_right_fn"](t_right)

    x_ic_np = np.random.uniform(*dom["x_range"], n_i)
    x_ic    = torch.tensor(x_ic_np, dtype=torch.float32, device=device).unsqueeze(1)
    t_ic    = torch.zeros(n_i, 1, device=device)
    u_ic    = config["initial_condition"](x_ic)

    return {
        "x_left":  x_left,  "t_left":  t_left,  "g_left":  g_left,
        "x_right": x_right, "t_right": t_right, "g_right": g_right,
        "x_ic":    x_ic,    "t_ic":    t_ic,    "u_ic":    u_ic,
    }


class PINN_Source(nn.Module):
    def __init__(self, hidden_u, hidden_f, activation="tanh", x_range=(0, 1), t_range=(0, 1)):
        super().__init__()
        self.register_buffer("x_min", torch.tensor(x_range[0], dtype=torch.float32))
        self.register_buffer("x_max", torch.tensor(x_range[1], dtype=torch.float32))
        self.register_buffer("t_min", torch.tensor(t_range[0], dtype=torch.float32))
        self.register_buffer("t_max", torch.tensor(t_range[1], dtype=torch.float32))

        def build_net(hidden, in_dim=2, out_dim=1):
            layers, dim = [], in_dim
            for h in hidden:
                layers += [nn.Linear(dim, h), make_activation(activation)]
                dim = h
            layers.append(nn.Linear(dim, out_dim))
            return nn.Sequential(*layers)

        self.subnet_u = build_net(hidden_u)
        self.subnet_f = build_net(hidden_f)

        for net in [self.subnet_u, self.subnet_f]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)

    def _normalize(self, x, t):
        x_n = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_n = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return x_n, t_n

    def forward_u(self, x, t):
        x_n, t_n = self._normalize(x, t)
        return self.subnet_u(torch.cat([x_n, t_n], dim=1))

    def forward_f(self, x, t):
        x_n, t_n = self._normalize(x, t)
        return self.subnet_f(torch.cat([x_n, t_n], dim=1))

    def forward(self, x, t):
        return self.forward_u(x, t), self.forward_f(x, t)


def compute_pde_residual(model, x, t, config):
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    if not t.requires_grad:
        t = t.detach().requires_grad_(True)

    u    = model.forward_u(x, t)
    f    = model.forward_f(x, t)
    u_t  = torch.autograd.grad(u,   t,   torch.ones_like(u),   create_graph=True)[0]
    u_x  = torch.autograd.grad(u,   x,   torch.ones_like(u),   create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x,   torch.ones_like(u_x), create_graph=True)[0]

    pde = config["pde"]
    L_u = pde["operator"](u, u_x, u_xx, pde["params"])
    return u_t - L_u - f


def _bc_side_loss(model, x_b, t_b, g_b, bc_type, normal_sign=1):
    if bc_type == "dirichlet":
        u_pred = model.forward_u(x_b, t_b)
        return torch.mean((u_pred - g_b) ** 2)
    else:
        x_b = x_b.detach().requires_grad_(True)
        u_pred = model.forward_u(x_b, t_b)
        du_dx  = torch.autograd.grad(
            u_pred, x_b,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True
        )[0]
        return torch.mean((normal_sign * du_dx - g_b) ** 2)


def compute_losses(model, data, config):

    bc = data["boundary"]

    residual = compute_pde_residual(model, data["x_col"], data["t_col"], config)
    loss_pde = torch.mean(residual ** 2)

    u_pred    = model.forward_u(data["x_obs"], data["t_obs"])
    loss_data = torch.mean((u_pred - data["u_obs"]) ** 2)

    u_ic_pred = model.forward_u(bc["x_ic"], bc["t_ic"])
    loss_ic   = torch.mean((u_ic_pred - bc["u_ic"]) ** 2)

    loss_bc_left  = _bc_side_loss(model, bc["x_left"],  bc["t_left"],
                                  bc["g_left"],  config["bc_left_type"],  normal_sign=-1)
    loss_bc_right = _bc_side_loss(model, bc["x_right"], bc["t_right"],
                                  bc["g_right"], config["bc_right_type"], normal_sign=+1)

    return {
        "pde":      loss_pde,
        "data":     loss_data,
        "ic":       loss_ic,
        "bc_left":  loss_bc_left,
        "bc_right": loss_bc_right,
    }


def total_loss_from_parts(parts, config):
    w_bc_left  = 10.0 if config["bc_left_type"]  == "neumann" else 5.0
    w_bc_right = 10.0 if config["bc_right_type"] == "neumann" else 5.0
    return (parts["pde"] +
            10.0 * parts["data"] +
            5.0  * parts["ic"] +
            w_bc_left  * parts["bc_left"] +
            w_bc_right * parts["bc_right"])


def evaluate_model(model, config, n_eval=80):
    model.eval()
    dom = config["domain"]
    x_plot = np.linspace(*dom["x_range"], n_eval)
    t_plot = np.linspace(*dom["t_range"], n_eval)
    X, T = np.meshgrid(x_plot, t_plot)

    F_exact = config["f_true"](X, T)
    U_exact = config["u_exact"](X, T)

    x_fl = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_fl = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        f_pred = model.forward_f(x_fl, t_fl).cpu().numpy().reshape(X.shape)
        u_pred = model.forward_u(x_fl, t_fl).cpu().numpy().reshape(X.shape)

    metrics_f = compute_metrics(f_pred, F_exact, x_plot, t_plot)
    metrics_u = compute_metrics(u_pred, U_exact, x_plot, t_plot)
    model.train()
    return {"f": metrics_f, "u": metrics_u}


def _make_history():
    return {"loss": [], "loss_pde": [], "loss_data": [], "loss_ic": [],
            "loss_bc_left": [], "loss_bc_right": []}


def _make_metrics_log():
    return {"epoch": [], "f_l2_rel": [], "f_h1_rel": [], "f_linf": [],
            "u_l2_rel": [], "u_h1_rel": []}


def _log_metrics(metrics_log, epoch_or_step, m, key="epoch"):
    metrics_log[key].append(epoch_or_step)
    metrics_log["f_l2_rel"].append(m["f"]["l2_rel"])
    metrics_log["f_h1_rel"].append(m["f"]["h1_rel"])
    metrics_log["f_linf"].append(m["f"]["linf"])
    metrics_log["u_l2_rel"].append(m["u"]["l2_rel"])
    metrics_log["u_h1_rel"].append(m["u"]["h1_rel"])


def train_adam(model, data, config):
    trn = config["training"]
    epochs      = trn["epochs"]
    print_every = trn["print_every"]
    eval_every  = trn["eval_every"]

    warmup_epochs = 0
    has_neumann   = (config["bc_left_type"] == "neumann" or
                     config["bc_right_type"] == "neumann")

    def _f_is_complex(config, n=20, threshold=2.0):
        dom = config["domain"]
        xs  = np.linspace(*dom["x_range"], n)
        ts  = np.linspace(*dom["t_range"], n)
        X, T = np.meshgrid(xs, ts)
        f_vals = config["f_true"](X, T)
        f_std  = np.std(f_vals)
        f_range = np.ptp(f_vals)
        return (f_range > threshold * f_std + 1e-8) or (f_range > 1.0)

    has_complex_f = _f_is_complex(config)
    needs_warmup  = has_neumann or has_complex_f

    if needs_warmup:
        reason = []
        if has_neumann:     reason.append("граничное условие Неймана")
        if has_complex_f:   reason.append("сложная/осциллирующая f")
        warmup_epochs = 2000
        print(f"Прогрев {warmup_epochs} эпох (subnet_f заморожена). Причина: {', '.join(reason)}.")

        for param in model.subnet_f.parameters():
            param.requires_grad_(False)

        opt_warmup = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=trn["learning_rate"]
        )
        sch_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_warmup, T_max=warmup_epochs, eta_min=1e-5
        )

        print(f"{'Эпоха':>8} | {'Loss':>10} | {'L_bc_L':>10} | {'L_bc_R':>10} | {'L_ic':>10} | {'u_L2%':>7}")
        print("-" * 65)

        for epoch in range(warmup_epochs):
            opt_warmup.zero_grad()
            parts = compute_losses(model, data, config)
            loss = (10.0 * parts["data"] +
                    5.0  * parts["ic"] +
                    (10.0 if config["bc_left_type"]  == "neumann" else 5.0) * parts["bc_left"] +
                    (10.0 if config["bc_right_type"] == "neumann" else 5.0) * parts["bc_right"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_warmup.step()
            sch_warmup.step()

            if epoch % 500 == 0 or epoch == warmup_epochs - 1:
                with torch.no_grad():
                    m = evaluate_model(model, config)
                u_l2 = m["u"]["l2_rel"] * 100
                print(f"{epoch:>8} | {loss.item():>10.2e} | "
                      f"{parts['bc_left'].item():>10.2e} | "
                      f"{parts['bc_right'].item():>10.2e} | "
                      f"{parts['ic'].item():>10.2e} | {u_l2:>6.2f}%")

        for param in model.subnet_f.parameters():
            param.requires_grad_(True)

        transition_epochs = 1000
        w_pde_transition  = 0.1
        lr_transition     = trn["learning_rate"] * 0.2

        opt_trans = torch.optim.Adam(model.parameters(), lr=lr_transition)
        sch_trans = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_trans, T_max=transition_epochs, eta_min=1e-6
        )

        print(f"\nПереходный этап {transition_epochs} эпох "
              f"(lr={lr_transition:.1e}, w_pde={w_pde_transition})...")
        print(f"{'Эпоха':>8} | {'Loss':>10} | {'L_pde':>10} | {'L_bc_L':>10} | {'L_bc_R':>10} | {'u_L2%':>7}")
        print("-" * 65)

        for epoch in range(transition_epochs):
            opt_trans.zero_grad()
            parts = compute_losses(model, data, config)
            w_bc_left  = 10.0 if config["bc_left_type"]  == "neumann" else 5.0
            w_bc_right = 10.0 if config["bc_right_type"] == "neumann" else 5.0
            loss = (w_pde_transition * parts["pde"] +
                    10.0 * parts["data"] +
                    5.0  * parts["ic"] +
                    w_bc_left  * parts["bc_left"] +
                    w_bc_right * parts["bc_right"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_trans.step()
            sch_trans.step()

            if epoch % 500 == 0 or epoch == transition_epochs - 1:
                with torch.no_grad():
                    m = evaluate_model(model, config)
                u_l2 = m["u"]["l2_rel"] * 100
                print(f"{epoch:>8} | {loss.item():>10.2e} | "
                      f"{parts['pde'].item():>10.2e} | "
                      f"{parts['bc_left'].item():>10.2e} | "
                      f"{parts['bc_right'].item():>10.2e} | {u_l2:>6.2f}%")

        print(f"\nПрогрев завершён. Запуск основного обучения ({epochs} эпох)...\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=trn["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history     = _make_history()
    metrics_log = _make_metrics_log()

    bc_info = f"ГУ: left={config['bc_left_type']:10s} | right={config['bc_right_type']}"
    print(bc_info)
    print(f"{'Эпоха':>8} | {'Loss':>10} | {'L_pde':>10} | {'L_data':>10} | "
          f"{'L_bc_L':>10} | {'L_bc_R':>10} | {'L_ic':>8} | {'f_L2%':>7} | {'u_L2%':>7}")
    print("-" * 110)

    for epoch in range(epochs):
        optimizer.zero_grad()
        parts = compute_losses(model, data, config)
        loss  = total_loss_from_parts(parts, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history["loss"].append(loss.item())
        for k in ["pde", "data", "ic", "bc_left", "bc_right"]:
            history[f"loss_{k}"].append(parts[k].item())

        if epoch % eval_every == 0 or epoch == epochs - 1:
            _log_metrics(metrics_log, epoch, evaluate_model(model, config))

        if epoch % print_every == 0 or epoch == epochs - 1:
            last = metrics_log
            if last["epoch"] and last["epoch"][-1] == epoch:
                f_str = f"{last['f_l2_rel'][-1]*100:7.2f}"
                u_str = f"{last['u_l2_rel'][-1]*100:7.2f}"
            else:
                f_str = u_str = "  —    "
            print(f"{epoch:>8} | {loss.item():>10.2e} | {parts['pde'].item():>10.2e} | "
                  f"{parts['data'].item():>10.2e} | {parts['bc_left'].item():>10.2e} | "
                  f"{parts['bc_right'].item():>10.2e} | {parts['ic'].item():>8.2e} | "
                  f"{f_str}% | {u_str}%")

    return history, metrics_log


def train_lbfgs(model, data, config):
    print("\nЗапуск L-BFGS...")
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=0.5, max_iter=20,
        history_size=100, line_search_fn="strong_wolfe")

    n_steps     = 300
    eval_every  = 100
    print_every = 100

    metrics_log = _make_metrics_log()
    metrics_log["step"] = metrics_log.pop("epoch")
    last_parts  = {}

    print(f"{'Шаг':>6} | {'Loss':>10} | {'L_pde':>10} | {'L_data':>10} | "
          f"{'L_bc_L':>10} | {'L_bc_R':>10} | {'L_ic':>8} | {'f_L2%':>7} | {'u_L2%':>7}")
    print("-" * 110)

    for step in range(n_steps):
        def closure():
            optimizer.zero_grad()
            d = {**data,
                 "x_col": data["x_col"].detach().requires_grad_(True),
                 "t_col": data["t_col"].detach().requires_grad_(True)}
            parts = compute_losses(model, d, config)
            loss = total_loss_from_parts(parts, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            last_parts.update({k: v.item() for k, v in parts.items()})
            last_parts["total"] = loss.item()
            return loss

        optimizer.step(closure)

        if step % eval_every == 0 or step == n_steps - 1:
            _log_metrics(metrics_log, step, evaluate_model(model, config), key="step")

        if step % print_every == 0 or step == n_steps - 1:
            last = metrics_log
            if last["step"] and last["step"][-1] == step:
                f_str = f"{last['f_l2_rel'][-1]*100:7.2f}"
                u_str = f"{last['u_l2_rel'][-1]*100:7.2f}"
            else:
                f_str = u_str = "  —    "
            print(f"{step:>6} | {last_parts.get('total', float('nan')):>10.2e} | "
                  f"{last_parts.get('pde', float('nan')):>10.2e} | "
                  f"{last_parts.get('data', float('nan')):>10.2e} | "
                  f"{last_parts.get('bc_left', float('nan')):>10.2e} | "
                  f"{last_parts.get('bc_right', float('nan')):>10.2e} | "
                  f"{last_parts.get('ic', float('nan')):>8.2e} | "
                  f"{f_str}% | {u_str}%")

    print("\nL-BFGS завершён.")
    return metrics_log


def plot_metrics_evolution(metrics_log, title_prefix="", step_key="epoch"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    steps = metrics_log[step_key]

    axes[0].semilogy(steps, np.array(metrics_log["f_l2_rel"])*100, 'b-o', label="f L2 rel %", markersize=3)
    axes[0].semilogy(steps, np.array(metrics_log["u_l2_rel"])*100, 'r-o', label="u L2 rel %", markersize=3)
    axes[0].set_xlabel(step_key); axes[0].set_ylabel("L2 relative error, %")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title("L2 relative error")

    axes[1].semilogy(steps, np.array(metrics_log["f_h1_rel"])*100, 'b-o', label="f H1 rel %", markersize=3)
    axes[1].semilogy(steps, np.array(metrics_log["u_h1_rel"])*100, 'r-o', label="u H1 rel %", markersize=3)
    axes[1].set_xlabel(step_key); axes[1].set_ylabel("H1 relative error, %")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_title("H1 relative error")

    plt.suptitle(f"{title_prefix} — Эволюция метрик", fontsize=13)
    plt.tight_layout(); plt.show()


def plot_results_extended(model, history, metrics_log, metrics_log_lbfgs, config, title_prefix="", save_dir=None):
    model.eval()
    dom    = config["domain"]
    n_plot = 80
    x_plot = np.linspace(*dom["x_range"], n_plot)
    t_plot = np.linspace(*dom["t_range"], n_plot)
    X, T   = np.meshgrid(x_plot, t_plot)

    F_exact = config["f_true"](X, T)
    U_exact = config["u_exact"](X, T)

    x_fl = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_fl = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        f_pred = model.forward_f(x_fl, t_fl).cpu().numpy().reshape(X.shape)
        u_pred = model.forward_u(x_fl, t_fl).cpu().numpy().reshape(X.shape)

    metrics_f = compute_metrics(f_pred, F_exact, x_plot, t_plot)
    metrics_u = compute_metrics(u_pred, U_exact, x_plot, t_plot)

    x_lo, x_hi = dom["x_range"]
    t_lo, t_hi = dom["t_range"]
    mx, mt = 0.1 * (x_hi - x_lo), 0.1 * (t_hi - t_lo)
    xi0 = np.searchsorted(x_plot, x_lo + mx)
    xi1 = np.searchsorted(x_plot, x_hi - mx, side="right")
    ti0 = np.searchsorted(t_plot, t_lo + mt)
    ti1 = np.searchsorted(t_plot, t_hi - mt, side="right")
    x_inner = x_plot[xi0:xi1]
    t_inner = t_plot[ti0:ti1]
    metrics_f_inner = compute_metrics(
        f_pred[ti0:ti1, xi0:xi1],
        F_exact[ti0:ti1, xi0:xi1],
        x_inner, t_inner,
    )

    pde_desc = config["pde"]["description"]
    pde_type = config["pde"]["type"]
    bc_desc  = f"ГУ: left={config['bc_left_type']} | right={config['bc_right_type']}"

    print(f"\n{'='*65}")
    print(f"{title_prefix}")
    print(f"PDE: {pde_desc}")
    print(f"{bc_desc}")
    print(f"{'='*65}")
    print_metrics(metrics_u, prefix="[u — полная область]")
    print_metrics(metrics_f, prefix="[f — полная область]")
    print_metrics(metrics_f_inner, prefix="[f — внутренняя]")

    f_error = np.abs(F_exact - f_pred)
    u_error = np.abs(U_exact - u_pred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, data_2d, cmap, ttl in [
        (axes[0,0], F_exact, "RdBu_r", "f(x,t) истинное"),
        (axes[0,1], f_pred,  "RdBu_r", "f(x,t) PINN"),
        (axes[0,2], f_error, "Reds",   f"Ошибка f: L2={metrics_f['l2_rel']*100:.1f}%"),
    ]:
        c = ax.contourf(X, T, data_2d, levels=50, cmap=cmap)
        plt.colorbar(c, ax=ax); ax.set_xlabel("x"); ax.set_ylabel("t"); ax.set_title(ttl)

    for tv in [0.0, 0.25, 0.5]:
        it = np.argmin(np.abs(t_plot - tv))
        axes[1,0].plot(x_plot, F_exact[it], "-",  label=f"t={tv} exact")
        axes[1,0].plot(x_plot, f_pred[it],  "--", label=f"t={tv} PINN")
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3); axes[1,0].set_title("Сечения f(x, t=const)")

    axes[1,1].semilogy(history["loss"], lw=0.8, label="total")
    axes[1,1].set_title("Loss"); axes[1,1].grid(alpha=0.3)

    for k, lbl in [("loss_pde", "PDE"), ("loss_data", "data"), ("loss_ic", "IC"),
                   ("loss_bc_left", f"BC-L ({config['bc_left_type'][0].upper()})"),
                   ("loss_bc_right", f"BC-R ({config['bc_right_type'][0].upper()})")]:
        axes[1,2].semilogy(history[k], label=lbl, lw=0.8)
    axes[1,2].legend(); axes[1,2].grid(alpha=0.3); axes[1,2].set_title("Компоненты потерь")

    plt.suptitle(f"{title_prefix} ({pde_type}, {bc_desc}) — источник f", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{title_prefix}_f.png",
                    dpi=300, bbox_inches="tight")

    plt.show()

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    for ax2, data_2d, cmap, ttl in [
        (axes2[0,0], U_exact, "viridis", "u(x,t) точное"),
        (axes2[0,1], u_pred,  "viridis", "u(x,t) PINN"),
        (axes2[0,2], u_error, "Reds",    f"Ошибка u: L2={metrics_u['l2_rel']*100:.1f}%"),
    ]:
        c = ax2.contourf(X, T, data_2d, levels=50, cmap=cmap)
        plt.colorbar(c, ax=ax2); ax2.set_xlabel("x"); ax2.set_ylabel("t"); ax2.set_title(ttl)

    for tv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        it = np.argmin(np.abs(t_plot - tv))
        axes2[1,0].plot(x_plot, U_exact[it], "-",  label=f"t={tv:.2f} exact")
        axes2[1,0].plot(x_plot, u_pred[it],  "--", label=f"t={tv:.2f} PINN")
    axes2[1,0].legend(fontsize=7); axes2[1,0].grid(alpha=0.3); axes2[1,0].set_title("Сечения u(x, t=const)")

    for xv in [0.25, 0.5, 0.75]:
        ix = np.argmin(np.abs(x_plot - xv))
        axes2[1,1].plot(t_plot, U_exact[:, ix], "-",  label=f"x={xv} exact")
        axes2[1,1].plot(t_plot, u_pred[:, ix],  "--", label=f"x={xv} PINN")
    axes2[1,1].legend(fontsize=7); axes2[1,1].grid(alpha=0.3); axes2[1,1].set_title("Сечения u(t, x=const)")

    axes2[1,2].axis("off")
    txt = (f"Метрики u(x,t)\n{'─'*28}\n"
           f"L2 relative:    {metrics_u['l2_rel']*100:.4f}%\n"
           f"H1 relative:    {metrics_u['h1_rel']*100:.4f}%\n"
           f"L∞ max error:   {metrics_u['linf']:.6f}\n"
           f"MAE:            {metrics_u['mae']:.6f}\n"
           f"RMSE:           {metrics_u['rmse']:.6f}")
    axes2[1,2].text(0.05, 0.95, txt, transform=axes2[1,2].transAxes,
                    fontsize=11, verticalalignment="top", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.suptitle(f"{title_prefix} ({pde_type}) — решение u", fontsize=13, y=1.01)
    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig2.savefig(f"{save_dir}/{title_prefix}_u.png",
                     dpi=300, bbox_inches="tight")

    plt.show()

    if metrics_log:
        plot_metrics_evolution(metrics_log, title_prefix + " [Adam]", step_key="epoch")
    if metrics_log_lbfgs:
        plot_metrics_evolution(metrics_log_lbfgs, title_prefix + " [L-BFGS]", step_key="step")

    return {"metrics_f_full": metrics_f, "metrics_u_full": metrics_u, "metrics_f_inner": metrics_f_inner}


def run_experiment(config_raw):
    config = prepare_config(config_raw)
    trn = config["training"]
    dom = config["domain"]

    print("=" * 65)
    print(f"PDE:  {config['pde']['description']}")
    print(f"ГУ:   left={config['bc_left_type']} | right={config['bc_right_type']}")
    print("Шаг 1/3: вычисление u_exact на сетке...")
    x_grid, t_grid, U = build_solution_grid(config)

    print("Шаг 2/3: генерация точек...")
    x_obs, t_obs, u_obs = generate_observations(x_grid, t_grid, U, config)
    x_col, t_col        = generate_collocation_points(config)
    boundary            = generate_boundary_points(config)

    data = {"x_obs": x_obs, "t_obs": t_obs, "u_obs": u_obs,
            "x_col": x_col, "t_col": t_col, "boundary": boundary}

    print("Шаг 3/3: обучение PINN...")
    print("=" * 65)

    model = PINN_Source(
        hidden_u   = trn["hidden_layers_u"],
        hidden_f   = trn["hidden_layers_f"],
        activation = trn["activation"],
        x_range    = dom["x_range"],
        t_range    = dom["t_range"],
    ).to(device)

    history, metrics_log  = train_adam(model, data, config)
    metrics_log_lbfgs     = train_lbfgs(model, data, config)

    return model, history, metrics_log, metrics_log_lbfgs, x_grid, t_grid, U, config

print("run_experiment — OK")


__all__ = [
    "run_experiment",
    "plot_results_extended",
    "plot_metrics_evolution",
    "make_consistent_config",
    "prepare_config",
    "device",
]
