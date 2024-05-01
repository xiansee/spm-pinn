import sys

sys.path.append("../")
import lightning.pytorch as pl
from core.data_module import DataModule
from core.dataset import SimulationDataset
from core.helper import plot
from core.loss import PINNLoss
from core.ocp_torch import get_graphite_ocp, get_nmc_ocp
from core.pinn_model import SPM_PINN
from core.physics_model import SPM
from core.training_module import TrainingModule
from torch import optim, vmap
import pickle

dataset = SimulationDataset(data_directory="../data")
data_module = DataModule(dataset=dataset, train_split=1, val_split=0, test_split=0)

model = SPM_PINN(
    Up=vmap(get_nmc_ocp),  # Positive electrode OCP as f(conc) [V]
    Cp_0=35263,  # Initial positive electrode Li concentration [mol/m3]
    Cp_max=63104,  # Max positive electrode Li concentration [mol/m3]
    Rp=5.22e-6,  # Positive electrode particle radius [m]
    ep_s=0.335,  # Positive electrode volume fraction [-]
    Lp=75.6e-6,  # Positive Electrode thickness [m]
    kp=5e-10,  # Positive electrode reaction rate constant [m^2.5/(mol^0.5.s)]
    Dp=1e-14,  # Positive electrode diffusivity [m2/s]
    Un=vmap(get_graphite_ocp),  # Negative electrode OCP as f(conc) [V]
    Cn_0=15528,  # Initial negative electrode Li concentration [mol/m3]
    Cn_max=33133,  # Max negative electrode Li concentration [mol/m3]
    Rn=5.22e-6,  # Negative electrode particle radius [m]
    en_s=0.335,  # Negative electrode volume fraction [-]
    Ln=75.6e-6,  # Negative Electrode thickness [m]
    kn=5e-10,  # Negative electrode reaction rate constant [m^2.5/(mol^0.5.s)]
    Dn=3e-14,  # Negative electrode diffusivity [m2/s]
    Ce=1000,  # Electrolyte Li concentration [mol/m3]
    R_cell=3.24e-4,  # Cell resistance [ohm m2])
    nn_hidden_size=20,
    nn_num_hidden_layers=8,
)
loss_fn = PINNLoss()
optimizer_algorithm = optim.LBFGS
optimizer_kwargs = {"lr": 0.01}
optimizer = optimizer_algorithm(params=model.parameters(), **optimizer_kwargs)

training_module = TrainingModule(
    model=model,
    loss_function=loss_fn,
    optimizer=optimizer,
)

trainer = pl.Trainer(
    # max_epochs=500,
    max_time={"seconds": 1800},
    logger=True,
    enable_progress_bar=False,
    enable_model_summary=False,
    enable_checkpointing=False,
)

trainer.fit(training_module, data_module)

I, Xp, Xn, Y, (N_t, N_rp, N_rn) = dataset[0]
out = model(I, Xp, Xn, N_t)

Cp0 = model.Cp[Xp[:, 0] == -1].detach().numpy()[:, 0]
Cn0 = model.Cn[Xn[:, 0] == -1].detach().numpy()[:, 0]

Cplast = model.Cp[Xp[:, 0] == 1].detach().numpy()[:, 0]
Cnlast = model.Cn[Xn[:, 0] == 1].detach().numpy()[:, 0]

Cp_surf = model.Cp[-N_t:].detach().numpy()[:, 0]
Cn_surf = model.Cn[-N_t:].detach().numpy()[:, 0]

with open("../data/spm0_Dp=1e-14_Dn=3e-14", "rb") as binary_file:
    true_data = pickle.load(binary_file)


# =========== Plots ================

plot(
    x=[
        true_data.get(SPM.rp_col)[0],
        true_data.get(SPM.rn_col)[0],
        true_data.get(SPM.rp_col)[0],
        true_data.get(SPM.rn_col)[0],
    ],
    y=[Cp0, Cn0, true_data.get(SPM.cp_col)[0], true_data.get(SPM.cn_col)[0]],
    series_names=[
        "Predicted positive",
        "Predicted negative",
        "True positive",
        "True negative",
    ],
    lines=True,
    markers=False,
    x_label=f"{SPM.rp_col} | {SPM.rn_col}",
    y_label=f"{SPM.cp_col} | {SPM.cn_col}",
    title="Initial concentration",
)

plot(
    x=[
        true_data.get(SPM.rp_col)[-1],
        true_data.get(SPM.rn_col)[-1],
        true_data.get(SPM.rp_col)[-1],
        true_data.get(SPM.rn_col)[-1],
    ],
    y=[Cplast, Cnlast, true_data.get(SPM.cp_col)[-1], true_data.get(SPM.cn_col)[-1]],
    series_names=[
        "Predicted positive",
        "Predicted negative",
        "True positive",
        "True negative",
    ],
    lines=True,
    markers=False,
    x_label=f"{SPM.rp_col} | {SPM.rn_col}",
    y_label=f"{SPM.cp_col} | {SPM.cn_col}",
    title="Final concentration",
)

plot(
    x=[true_data.get(SPM.time_col), true_data.get(SPM.time_col)],
    y=[true_data.get(SPM.cp_surf_col), Cp_surf],
    series_names=["True", "Predicted"],
    lines=True,
    markers=False,
    x_label=SPM.time_col,
    y_label=SPM.cp_surf_col,
    title="Positive electrode surface concentration",
)

plot(
    x=[true_data.get(SPM.time_col), true_data.get(SPM.time_col)],
    y=[true_data.get(SPM.cn_surf_col), Cn_surf],
    series_names=["True", "Predicted"],
    lines=True,
    markers=False,
    x_label=SPM.time_col,
    y_label=SPM.cp_surf_col,
    title="Negative electrode surface concentration",
)

plot(
    x=[true_data.get(SPM.time_col), true_data.get(SPM.time_col)],
    y=[true_data.get(SPM.voltage_col), out.detach().numpy()[:, 0]],
    series_names=["True", "Predicted"],
    lines=True,
    markers=False,
    x_label=SPM.time_col,
    y_label=SPM.voltage_col,
    title="Terminal Voltage",
)
