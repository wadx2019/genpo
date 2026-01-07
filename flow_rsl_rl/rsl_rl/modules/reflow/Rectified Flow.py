import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
from flow import RectifiedFlowAffine, RectifiedFlowMLP, RectifiedFlowGlow
from flow import MLP, AffineRectifiedFlow, MultiScaleGlow

###device####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

D = 10.
M = D+5
VAR = 0.3
DOT_SIZE = 4
COMP = 3
####data generation

# initial_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
# initial_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., D / 2.], [-D * np.sqrt(3) / 2., D / 2.], [0.0, - D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
# initial_model = MixtureSameFamily(initial_mix, initial_comp)
# samples_0 = initial_model.sample([10000])

samples_0 = torch.randn(10000, 2, device=device)

target_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)], device=device))
target_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D * np.sqrt(3) / 2.]], device=device).float(), VAR * torch.stack([torch.eye(2, device=device) for i in range(COMP)]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample([10000])
print('Shape of the samples:', samples_0.shape, samples_1.shape)

plt.figure(figsize=(4,4))
plt.xlim(-M,M)
plt.ylim(-M,M)
plt.title(r'Samples from $\pi_0$ and $\pi_1$')
plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(), alpha=0.1, label=r'$\pi_1$')
plt.legend()

plt.show()

def train_rectified_flow(rectified_flow, optimizer, pairs, batchsize, inner_iters):
    loss_curve = []
    for i in range(inner_iters + 1):
        optimizer.zero_grad()
        indices = torch.randperm(len(pairs))[:batchsize]
        batch = pairs[indices]
        z0 = batch[:, 0].detach().clone()
        z1 = batch[:, 1].detach().clone()
        z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)

        pred,_ = rectified_flow.model(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()

        optimizer.step()
        loss_curve.append(np.log(loss.item()))  ## to store the loss curve

    return rectified_flow, loss_curve


@torch.no_grad()
def draw_plot(rectified_flow, z0, z1, N=None):
    traj,Jacbians = rectified_flow.sample_ode(z0=z0)

    plt.figure(figsize=(4, 4))
    plt.xlim(-M, M)
    plt.ylim(-M, M)

    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
    plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
    plt.legend()
    plt.title('Distribution')
    plt.tight_layout()

    traj_particles = torch.stack(traj)
    plt.figure(figsize=(4, 4))
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.axis('equal')
    for i in range(30):
        plt.plot(traj_particles[:, i, 0].cpu().numpy(), traj_particles[:, i, 1].cpu().numpy())
    plt.title('Transport Trajectory')
    plt.show()

x_0 = samples_0.detach().clone()[torch.randperm(len(samples_0))]
x_1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]
x_pairs = torch.stack([x_0, x_1], dim=1)
print(x_pairs.shape)

iterations = 10000
batchsize = 2048
input_dim = 2

################## Normal MLP #################################################
rectified_flow_1 = RectifiedFlowAffine(model=AffineRectifiedFlow(input_dim).to(device), num_steps=5)
optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)
rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer, x_pairs, batchsize, iterations)


plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
plt.title('Training Loss Curve')
plt.show()

draw_plot(rectified_flow_1, z0=torch.randn(2000, 2, device=device), z1=samples_1.detach().clone(), N=10)

def draw_density(rectified_flow, z0, z1, N=None):
    n_samples = z0.shape[0]

    # 生成双峰分布的数据
    data1 = np.random.multivariate_normal(mean=[1, 1], cov=[[1, 0.5], [0.5, 1]], size=n_samples // 2)
    data2 = np.random.multivariate_normal(mean=[-1, -1], cov=[[1, -0.3], [-0.3, 1]], size=n_samples // 2)
    positions = np.vstack([data1, data2])

    # 计算每个点的概率密度
    kde = gaussian_kde(positions.T)
    density_values = kde(positions.T)

    # 绘制热力图
    plt.figure(figsize=(10, 8))

    # 方法1: 使用hexbin
    # plt.subplot(2, 2, 1)
    # plt.hexbin(positions[:, 0], positions[:, 1], C=density_values, gridsize=30, cmap='viridis')
    # plt.colorbar(label='Probability Density')
    # plt.title('Hexbin Plot with Density Values')
    # plt.xlabel('X position')
    # plt.ylabel('Y position')

    # 方法2: 使用散点图+透明度
    plt.subplot(2, 2, 2)
    plt.scatter(positions[:, 0], positions[:, 1], c=density_values, s=10, alpha=0.5, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.title('Scatter Plot with Density Values')
    plt.xlabel('X position')
    plt.ylabel('Y position')


    # 方法4: 使用2D直方图
    # plt.subplot(2, 2, 4)
    # plt.hist2d(positions[:, 0], positions[:, 1], bins=30, cmap='viridis')
    # plt.colorbar(label='Counts')
    # plt.title('2D Histogram')
    # plt.xlabel('X position')
    # plt.ylabel('Y position')

    plt.tight_layout()
    plt.show()