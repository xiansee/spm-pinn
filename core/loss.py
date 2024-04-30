import torch
from torch import Tensor, mean, nn, square


class PINNLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.w1 = 1
        self.w2 = 1
        self.w3 = 1 / 10
        self.w4 = 1
        self.w5 = 1

    def forward(
        self,
        V_pred: Tensor,
        V_true: Tensor,
        Cp: Tensor,
        Cn: Tensor,
        Xp: Tensor,
        Xn: Tensor,
        Dp,
        Dn,
        Jp,
        Jn,
        N_t: int,
        N_r: int = None,
    ) -> Tensor:
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

        dCp_dt = Cp_grad[:, 0]
        dCp_dr = Cp_grad[:, 1]
        dCn_dt = Cn_grad[:, 0]
        dCn_dr = Cn_grad[:, 1]
        tp = Xp[:, 0]
        rp = Xp[:, 1]
        tn = Xn[:, 0]
        rn = Xn[:, 1]

        ITp = square(rp) * Dp * dCp_dr  # intermediate term
        ITp_grad = torch.autograd.grad(
            outputs=ITp,
            inputs=Xp,
            grad_outputs=torch.ones_like(ITp),
            retain_graph=True,
            create_graph=True,
        )[0]
        dITp_dr = ITp_grad[:, 1]

        ITn = square(rn) * Dn * dCn_dr  # intermediate term
        ITn_grad = torch.autograd.grad(
            outputs=ITn,
            inputs=Xn,
            grad_outputs=torch.ones_like(ITn),
            retain_graph=True,
            create_graph=True,
        )[0]
        dITn_dr = ITn_grad[:, 1]

        Cp_max = 63104  # max(Cp)
        Cp_min = 1000  # min(Cp)
        Cn_max = 33133  # max(Cn)
        Cn_min = 1000  # min(Cn)
        V_max = 4.2  # max(V_pred)
        V_min = 2.5  # min(V_pred)
        Cp0 = Cp[Xp[:, 0] == 0]
        Cn0 = Cn[Xn[:, 0] == 0]
        Cp0_true = 35263
        Cn0_true = 15527

        Rp = 5.22e-6 / 10e-6  # Positive electrode particle radius [m]
        Rn = 5.86e-6 / 10e-6

        P1 = mean(square(dCp_dt - (1 / square(rp) * dITp_dr)))
        P2 = mean(square(dCp_dr[:N_t]))
        P3 = mean(square(Dp * 10e6 * dCp_dr[-N_t:] + Jp))
        P4 = mean(square(Cp0 - Cp0_true))

        N1 = mean(square(dCn_dt - (1 / square(rn) * dITn_dr)))
        N2 = mean(square(dCn_dr[:N_t]))
        N3 = mean(square(Dn * 10e6 * dCn_dr[-N_t:] + Jn))
        N4 = mean(square(Cn0 - Cn0_true))

        V_loss = mse_loss_fn(V_pred, V_true)

        s1_p = square(max(tp)) / (Cp_max - Cp_min) ** 2
        s2_p = Rp / (Cp_max - Cp_min) ** 2
        s3_p = 1 / Jp**2
        s4_p = 1 / (Cp0_true)

        s1_n = square(max(tn)) / (Cn_max - Cn_min) ** 2
        s2_n = Rn / (Cn_max - Cn_min) ** 2
        s3_n = 1 / Jn**2
        s4_n = 1 / (Cn0_true)
        s5 = 1 / (V_max - V_min) ** 2

        # print(
        #     f"P1={(self.w1 * s1_p * P1).item():.5E}, N1={(self.w1 * s1_n * N1).item():.5E}, P2={(self.w2 * s2_p * P2).item():.5E}, N2={(self.w2 * s2_n * N2).item():.5E}, P3={(self.w3 * s3_p * P3).item():.5E}, N3={(self.w3 * s3_n * N3).item():.5E}, P4={(self.w4 * s4_p * P4).item():.5E}, N4={(self.w4 * s4_n * N4).item():.5E}, Vloss={(self.w5 * s5 * V_loss).item():.5E}"
        # )
        return (
            # self.w1 * s1_p * P1
            # + self.w1 * s1_n * N1
            # + self.w2 * s2_p * P2
            # + self.w2 * s2_n * N2
            +self.w3 * s3_p * P3
            + self.w3 * s3_n * N3
            + self.w4 * s4_p * P4
            + self.w4 * s4_n * N4
            + self.w5 * s5 * V_loss
        )


class RMSE(nn.Module):
    """Root mean square error loss function."""

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        mse_loss_fn = nn.MSELoss()
        rmse = torch.sqrt(mse_loss_fn(y_pred, y_true))
        return rmse
