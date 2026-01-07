import torch
import math
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
# import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from torch.func import jacrev, vmap
from torch.distributions import Normal

class MLP_w_jac(nn.Module):
    """Multi-layer Perceptron with Jacobian computation capability.

    Args:
        input_dim (int): Dimension of input features. Default: 2
        hidden_num (int): Number of hidden units in each layer. Default: 256
        output_dim (int): Dimension of output features. Default: 2
    """

    def __init__(self, input_dim: int = 2, hidden_num: int = 256, output_dim: int = 2, activation: torch.nn.Module = torch.nn.Tanh()) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc4 = nn.Linear(hidden_num, output_dim, bias=True)
        self.activation = activation


    def forward(self,
                x_input: torch.Tensor,
                observations: torch.Tensor,
                t: torch.Tensor,
                jac: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the MLP.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            observations (torch.Tensor): Observation tensor of shape (batch_size, 1)
            t (torch.Tensor): Time tensor of shape (batch_size, 1)
            jac (bool): Whether to compute Jacobian. Default: True

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output tensor of shape (batch_size, output_dim)
                - Jacobian tensor if jac=True, else None
        """
        # Combine inputs
        inputs = torch.cat([x_input, observations, t], dim=1)

        # Forward pass through the network
        x = self.activation(self.fc1(inputs))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        output = self.fc4(x) * 0.5

        # Compute Jacobian if requested
        J = self._compute_jacobian(x_input, observations, t) if jac else None

        return output, J

    def _model_func(self,
                    x_in: torch.Tensor,
                    observations: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        """Helper function for Jacobian computation.

        Args:
            x_in (torch.Tensor): Single input sample
            observations (torch.Tensor): Single observation sample
            t (torch.Tensor): Single time sample

        Returns:
            torch.Tensor: Output for single sample
        """
        combined = torch.cat([x_in, observations, t], dim=0).unsqueeze(0)
        x = self.activation(self.fc1(combined))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return (self.fc4(x) * 0.5).squeeze(0)

    def _compute_jacobian(self,
                          x_input: torch.Tensor,
                          observations: torch.Tensor,
                          t: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian of the model output with respect to input.

        Args:
            x_input (torch.Tensor): Input tensor
            observations (torch.Tensor): Observation tensor
            t (torch.Tensor): Time tensor

        Returns:
            torch.Tensor: Jacobian tensor
        """
        jacobian_fn = jacrev(self._model_func, argnums=0)
        batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0, 0))
        return batched_jacobian_fn(x_input, observations, t)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    """Multi-layer Perceptron with Jacobian computation capability.

    Args:
        input_dim (int): Dimension of input features. Default: 2
        hidden_num (int): Number of hidden units in each layer. Default: 256
        output_dim (int): Dimension of output features. Default: 2
    """

    def __init__(self, input_dim: int = 2, hidden_dim: list = [256, 256, 256], output_dim: int = 2, t_dim = 32, activation: torch.nn.Module = torch.nn.Tanh()) :
        super().__init__()
        assert len(hidden_dim) > 0, "hidden_dim list must not be empty"

        layers = []
        layers.append(nn.Linear(input_dim + t_dim, hidden_dim[0], bias=True))
        layers.append(activation)

        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=True))
            layers.append(activation)

        layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=True))
        self.net = nn.Sequential(*layers)





    def forward(self,
                x_input: torch.Tensor,
                observations: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x_input (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            observations (torch.Tensor): Observation tensor of shape (batch_size, 1)
            t (torch.Tensor): Time tensor of shape (batch_size, 1)


        Returns:
            torch.Tensor:
                - Output tensor of shape (batch_size, output_dim)
        """
        # Combine inputs
        # print(x_input.shape, observations.shape, t.shape)
        inputs = torch.cat([x_input, observations, t], dim=1)
        output = self.net(inputs)

        return output

class Flow(nn.Module):
    def __init__(self, input_dim, output_dim, a_dim, actor_hidden_dim, time_dim, time_hidden_dim, activation, N, p, device):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.a_dim = a_dim
        self.N = N
        self.dt = 1/N
        self.p = p
        self.vec_field = MLP(input_dim=self.input_dim, hidden_dim=actor_hidden_dim, output_dim=self.output_dim, t_dim=time_dim, activation=activation)
        self.device = device
        self.dist  = Normal(torch.zeros(self.a_dim *2, device=device), torch.ones(self.a_dim * 2, device=device))

        ## time embedding
        layers = [SinusoidalPosEmb(time_dim)]
        in_dim = time_dim
        for hidden_dim in time_hidden_dim:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            layers.append(activation)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, time_dim))

        self.time_mlp = nn.Sequential(*layers)
        # t = torch.arange(self.N, device=self.device)/self.N
        # self.t_em = self.time_mlp(t)

    def _Heun_method(self, observations,  x):
        with torch.enable_grad():
            observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
            num_envs = observations.shape[0]

            z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
            y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]


            for i in range(self.N):

                t = torch.ones((num_envs, ), device=self.device) * i / self.N
                t_em = self.time_mlp(t)


                z_transformed = self.vec_field(y, observations, t_em)
                z_in = z + z_transformed * self.dt

                y_transformed = self.vec_field(z_in, observations, t_em)
                y_in = y + y_transformed * self.dt
                z = self.p * z_in + (1-self.p) * y_in
                y = self.p * y_in + (1-self.p) * z

            out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def _Heun_method_2(self, observations,  x):


        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        num_envs = observations.shape[0]

        z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
        y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]

        for i in range(self.N):

            t = torch.ones((num_envs, ), device=self.device) * i / self.N
            t_em = self.time_mlp(t)

            z_transformed = self.vec_field(y, observations, t_em)
            z_in = z + z_transformed * self.dt

            y_transformed = self.vec_field(z_in, observations, t_em)
            y_in = y + y_transformed * self.dt
            z = self.p * z_in + (1-self.p) * y_in
            y = self.p * y_in + (1-self.p) * z

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def _Heun_method_inverse(self, observations,  x):

        observations = observations.unsqueeze(0) if observations.dim() == 1 else observations
        num_envs = observations.shape[0]

        z = x[..., :self.a_dim].unsqueeze(0) if x.dim() == 1 else x[..., :self.a_dim]
        y = x[..., self.a_dim:].unsqueeze(0) if x.dim() == 1 else x[..., self.a_dim:]
        for i in reversed(range(self.N)):

            t = torch.ones((num_envs, ), device=self.device) * i / self.N
            t_em = self.time_mlp(t)

            y_in = (y - (1 - self.p) * z)/self.p
            z_in = (z - (1 - self.p) * y_in)/self.p
            y_transformed = self.vec_field(z_in, observations, t_em)
            y = y_in - y_transformed * self.dt
            z_transformed = self.vec_field(y, observations, t_em)
            z = z_in - z_transformed * self.dt

        out = torch.cat([z, y], dim=-1)

        return out.squeeze(0)

    def forward(self, observations, jac):
        batch = 4096
        num_envs = observations.shape[0]

        action_aug_0 = torch.randn(num_envs, self.a_dim * 2, device=self.device)
        log_probs = self.dist.log_prob(action_aug_0).sum(dim=-1)
        action_aug = torch.empty_like(action_aug_0)

        n = num_envs // batch
        n += int(num_envs % batch != 0)

        for i in range(n):
            action_aug[i*batch:min((i+1)*batch, num_envs)] = self._Heun_method(observations[i*batch:min((i+1)*batch, num_envs)],  action_aug_0[i*batch:min((i+1)*batch, num_envs)])
            # breakpoint()
            if jac:
                # assert action_aug_0.requires_grad, "input requires grad"
                jacobian_fn = jacrev(self._Heun_method, argnums = 1)
                batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0,  0))
                J = batched_jacobian_fn(observations[i*batch:min((i+1)*batch, num_envs)], action_aug_0[i*batch:min((i+1)*batch, num_envs)]).detach()

                log_probs[i*batch:min((i+1)*batch, num_envs)] = log_probs[i*batch:min((i+1)*batch, num_envs)] - torch.log(torch.abs(torch.linalg.det(J)))

        return action_aug, log_probs

    def inference(self, observations):
        batch = 4096
        num_envs = observations.shape[0]

        action_aug_0 = torch.randn(num_envs, self.a_dim * 2, device=self.device)
        action_aug = torch.empty_like(action_aug_0)

        n = num_envs // batch
        n += int(num_envs % batch != 0)

        for i in range(n):
            action_aug[i*batch:min((i+1)*batch, num_envs)] = self._Heun_method_2(observations[i*batch:min((i+1)*batch, num_envs)],  action_aug_0[i*batch:min((i+1)*batch, num_envs)])

        return action_aug

    def inverse(self, observations, action_aug,  jac=False):

        action_aug_0 = self._Heun_method_inverse(observations, action_aug)
        log_probs = self.dist.log_prob(action_aug_0).sum(dim=-1)

        if jac:

            jacobian_fn = jacrev(self._Heun_method_inverse, argnums=1)
            batched_jacobian_fn = vmap(jacobian_fn, in_dims=(0, 0))
            J = batched_jacobian_fn(observations, action_aug)

            log_probs = log_probs + torch.log(torch.abs(torch.linalg.det(J)))

        return log_probs





