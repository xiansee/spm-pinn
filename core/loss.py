import torch
from torch import Tensor, mean, nn, square


class PINNLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = 2
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1
        self.w5 = 10

    def forward(
        self, V_pred: Tensor, V_true: Tensor, Xp: Tensor, Xn: Tensor, N_t: int, model
    ) -> Tensor:
        Cp, Cp0_true, Cp_max = model.Cp, model.Cp_0, model.Cp_max
        Cn, Cn0_true, Cn_max = model.Cn, model.Cn_0, model.Cn_max
        jp, jn = model.jp, model.jn
        Dp, Dn = model.Dp, model.Dn
        Rp, Rn = model.Rp, model.Rn

        mse_loss_fn = nn.MSELoss()

        Cp_grad = torch.autograd.grad(
            outputs=Cp,
            inputs=Xp,
            grad_outputs=torch.ones_like(Cp),
            retain_graph=True,
            create_graph=True,
        )[0]
        Cn_grad = torch.autograd.grad(
            outputs=Cn,
            inputs=Xn,
            grad_outputs=torch.ones_like(Cn),
            retain_graph=True,
            create_graph=True,
        )[0]

        dCp_dt = Cp_grad[:, 0] / 50
        dCp_dr = Cp_grad[:, 1] / 6e-6
        dCn_dt = Cn_grad[:, 0] / 50
        dCn_dr = Cn_grad[:, 1] / 6e-6

        tp = model.unnormalize_data(Xp[:, 0], 50)
        rp = model.unnormalize_data(Xp[:, 1], 6e-6)
        tn = model.unnormalize_data(Xn[:, 0], 50)
        rn = model.unnormalize_data(Xn[:, 1], 6e-6)

        ITp = square(rp) * Dp * dCp_dr  # intermediate term
        ITp_grad = torch.autograd.grad(
            outputs=ITp,
            inputs=Xp,
            grad_outputs=torch.ones_like(ITp),
            retain_graph=True,
            create_graph=True,
        )[0]
        dITp_dr = ITp_grad[:, 1] / 6e-6

        ITn = square(rn) * Dn * dCn_dr  # intermediate term
        ITn_grad = torch.autograd.grad(
            outputs=ITn,
            inputs=Xn,
            grad_outputs=torch.ones_like(ITn),
            retain_graph=True,
            create_graph=True,
        )[0]
        dITn_dr = ITn_grad[:, 1] / 6e-6

        Cp_max, Cp_min = max(Cp), min(Cp)
        Cn_max, Cn_min = max(Cn), min(Cn)
        V_max, V_min = max(V_pred), min(V_pred)

        Cp0 = Cp[Xp[:, 0] == -1]
        Cn0 = Cn[Xn[:, 0] == -1]

        t1_p = mean(square(dCp_dt - (1 / square(rp) * dITp_dr)))
        t2_p = mean(square(dCp_dr[:N_t]))
        t3_p = mean(square(Dp * dCp_dr[-N_t:] + jp))
        t4_p = mean(square(Cp0 - Cp0_true))

        t1_n = mean(square(dCn_dt - (1 / square(rn) * dITn_dr)))
        t2_n = mean(square(dCn_dr[:N_t]))
        t3_n = mean(square(Dn * dCn_dr[-N_t:] + jn))
        t4_n = mean(square(Cn0 - Cn0_true))

        t5 = mse_loss_fn(V_pred, V_true)

        s1_p = square(max(tp)) / (Cp_max - Cp_min) ** 2
        s2_p = Rp**2 / (Cp_max - Cp_min) ** 2
        s3_p = 1 / max(square(jp))
        s4_p = 1 / (Cp_max - Cp_min) ** 2

        s1_n = square(max(tn)) / (Cn_max - Cn_min) ** 2
        s2_n = Rn**2 / (Cn_max - Cn_min) ** 2
        s3_n = 1 / max(square(jn))
        s4_n = 1 / (Cn_max - Cn_min) ** 2
        s5 = 1 / (V_max - V_min) ** 2

        print(
            f"t1_p={(self.w1 * s1_p * t1_p).item():.2E}, t1_n={(self.w1 * s1_n * t1_n).item():.2E}, t2_p={(self.w2 * s2_p * t2_p).item():.2E}, t2_n={(self.w2 * s2_n * t2_n).item():.2E}, t3_p={(self.w3 * s3_p * t3_p).item():.2E}, t3_n={(self.w3 * s3_n * t3_n).item():.2E}, t4_p={(self.w4 * s4_p * t4_p).item():.2E}, t4_n={(self.w4 * s4_n * t4_n).item():.2E}, t5={(self.w5 * s5 * t5).item():.2E}"
        )

        return (
            self.w1 * s1_p * t1_p
            + self.w1 * s1_n * t1_n
            + self.w2 * s2_p * t2_p
            + self.w2 * s2_n * t2_n
            + self.w3 * s3_p * t3_p
            + self.w3 * s3_n * t3_n
            + self.w4 * s4_p * t4_p
            + self.w4 * s4_n * t4_n
            + self.w5 * s5 * t5
        )


class RMSE(nn.Module):
    """Root mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_loss_fn = nn.MSELoss()
        rmse = torch.sqrt(mse_loss_fn(y_pred, y_true))
        return rmse
