import torch
from torch import nn
import torch.nn.functional as F
from utils.gumbel_softmax import gumbel_softmax # 假设你保留了 utils/gumbel_softmax.py

class CausalGraphLearner(nn.Module):
    """
    封装因果图学习逻辑 (G 和 GT)
    """
    def __init__(self, n_nodes, n_groups, group_policy, gumbel_tau_start):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_groups = n_groups
        self.group_policy = group_policy # e.g., "mul_2_every_10"
        self.gumbel_tau = gumbel_tau_start
        
        # 注册 G (group_assignment_matrix) 为 buffer (非训练参数)
        self.register_buffer("group_assignment_matrix", torch.zeros(n_nodes, n_groups))
        
        # 注册 GT (learnable_grouped_adj) 为 Parameter (可训练)
        # 注意：这里我们使用 torch.rand 而不是 0.5 来初始化，以便学习
        # 原始代码在 'train' 循环中初始化 GT，这里我们在 init 中完成
        initial_gt = torch.rand(n_groups, n_nodes) * 0.1 + 0.45 # ~0.5
        self.learnable_grouped_adj = nn.Parameter(initial_gt)
        
        self.update_groups(epoch=0) # 初始化 G 和 GT

    def get_effective_adj_matrix(self):
        """
        计算 G * GT (sigmoid) 得到 (N, N) 的有效邻接矩阵
        这是原代码中的 'Graph'
        """
        # G: (N, G_n), GT: (G_n, N)
        # 注意：原代码使用 sigmoid(GT)。我们将这个逻辑保留在这里。
        return self.group_assignment_matrix @ torch.sigmoid(self.learnable_grouped_adj)

    def sample_graph(self, effective_adj, batch_size, mode='gumbel'):
        """
        从有效邻接矩阵中采样
        :param effective_adj: (N, N) 矩阵
        :param batch_size: B
        :param mode: 'gumbel' (用于 graph_discov) 或 'bernoulli' (用于 data_pred)
        """
        if mode == 'bernoulli':
            # 原 'sample_bernoulli'
            sample_matrix = effective_adj.unsqueeze(0).expand(batch_size, -1, -1)
            return torch.bernoulli(sample_matrix).float()
            
        elif mode == 'gumbel':
            # 原 'gumbel_sigmoid_sample'
            # (N, N) -> (B, N, N, 1)
            prob = effective_adj.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
            # (B, N, N, 2)
            logits = torch.cat([prob, (1.0 - prob)], dim=-1)
            # (B, N, N)
            samples = gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)[..., 0]
            return samples
        else:
            raise ValueError(f"Unknown sample mode: {mode}")

    def update_groups(self, epoch):
        if self.group_policy is None or self.group_policy == "None":
            if epoch == 0:
                self.n_groups = self.n_nodes

                # 初始化 G = I
                new_g = torch.eye(self.n_nodes, device=self.learnable_grouped_adj.device)

                with torch.no_grad():
                    self.group_assignment_matrix = new_g
                    self.learnable_grouped_adj.data = (torch.ones((self.n_groups, self.n_nodes)) * 0.5).to(self.learnable_grouped_adj.device)
            return False

        group_mul = int(self.group_policy.split("_")[1])
        group_every = int(self.group_policy.split("_")[3])

        if epoch % group_every == 0 and (self.n_groups < self.n_nodes or epoch == 0):

            # -------- 更新 GT --------
            if epoch != 0:
                old_n_groups = self.n_groups
                self.n_groups = min(self.n_groups * group_mul, self.n_nodes)

                new_gt = torch.sigmoid(self.learnable_grouped_adj).detach().cpu().repeat_interleave(group_mul, 0)[:self.n_groups, :]
                new_gt = 1.0 - (1.0 - new_gt)**(1.0 / group_mul)

                with torch.no_grad():
                    self.learnable_grouped_adj.data = new_gt.to(self.learnable_grouped_adj.device)

            # -------- 更新 G --------
            new_g = torch.zeros(self.n_nodes, self.n_groups, device=self.learnable_grouped_adj.device)
            nodes_per_group = self.n_nodes // self.n_groups
            
            idx = 0
            for g in range(self.n_groups):
                for _ in range(nodes_per_group):
                    if idx >= self.n_nodes:
                        break
                    new_g[idx, g] = 1
                    idx += 1

            while idx < self.n_nodes:
                new_g[idx, self.n_groups - 1] = 1
                idx += 1

            with torch.no_grad():
                self.group_assignment_matrix = new_g  # ✅ 不再使用 register_buffer

            print(f"Epoch {epoch}: Updated groups to {self.n_groups}")
            return True

        return False

    
    


def test_causal_graph():

    print("\n====== Test: CausalGraphLearner ======")

    n_nodes = 8
    n_groups = 2
    group_policy = "mul_2_every_1"
    tau = 0.5

    model = CausalGraphLearner(
        n_nodes=n_nodes,
        n_groups=n_groups,
        group_policy=group_policy,
        gumbel_tau_start=tau
    )

    print("[Init] G shape:", model.group_assignment_matrix.shape)
    print("[Init] GT shape:", model.learnable_grouped_adj.shape)

    # 1. 获取有效邻接矩阵
    A = model.get_effective_adj_matrix()
    print("[Adj] Effective A shape:", A.shape)
    assert A.shape == (n_nodes, n_nodes)

    # 2. gumbel 采样测试
    sampled = model.sample_graph(A, batch_size=4, mode='gumbel')
    print("[Sample] Gumbel sample:", sampled.shape)
    assert sampled.shape == (4, n_nodes, n_nodes)

    # 3. bernoulli
    bern_sample = model.sample_graph(A, batch_size=3, mode='bernoulli')
    print("[Sample] Bernoulli sample:", bern_sample.shape)
    assert bern_sample.shape == (3, n_nodes, n_nodes)

    # 4. 测试一次 group update
    changed = model.update_groups(epoch=1)
    print("[Update] group updated:", changed)
    print("[Update] new G shape:", model.group_assignment_matrix.shape)
    print("[Update] new GT shape:", model.learnable_grouped_adj.shape)

    assert model.group_assignment_matrix.shape[1] <= n_nodes

    print("✅ All test passed.")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_causal_graph()