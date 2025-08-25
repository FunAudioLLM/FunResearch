import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class SequentialDPClustering(nn.Module):
    def __init__(self, k=10, beta=0.2, threshold=0.7, use_dynamic_threshold=False, max_span=4):
        super().__init__()
        self.k = k
        self.beta = beta
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.max_span = max_span

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, D, T] — encoder 输出

        Returns:
            cluster_embs_padded: [B, D, T'] — 聚类后的 token（padding 对齐）
            cluster_lengths: List[List[int]] — 每个 cluster 对应帧数
        """
        B, D, T = x.size()
        x_t = x.transpose(1, 2)  # [B, T, D]

        # Step 1: 相似度矩阵 φ(xi, xj) — [B, T, T]
        sim = self._pairwise_similarity(x_t)
        # Step 2: 局部密度 ρ_i — [B, T]
        rho = self._local_density(sim)
        # Step 3: 最小距离 δ_i（不能完全批量化） — [B, T]
        delta = self._delta(sim, rho)
        # Step 4: s_i = ρ_i * δ_i — [B, T]
        s = rho * delta

        # Step 5: 开始聚类
        cluster_embs_list = []
        cluster_lengths_list = []

        for b in range(B):
            x_b = x_t[b]           # [T, D]
            sim_b = sim[b]         # [T, T]
            s_b = s[b]             # [T]

            # pdb.set_trace()
            dynamic_threshold = self.threshold
            if self.use_dynamic_threshold:
                # dynamic_threshold = sim_b.mean().item()
                mu = sim_b.mean().item()
                sigma = sim_b.std().item()
                # 抑制 token 数下降过快的问题
                dynamic_threshold = mu + 0.5 * sigma  # 0.5 可微调为 0.3~1.0
                # pdb.set_trace()

            # pdb.set_trace()
            # === 缓存 sim_score [T, T] ===
            sim_score = sim_b - self.beta * s_b.view(1, -1)

            # === 贪心聚类过程 ===
            assigned = torch.zeros(T, dtype=torch.bool, device=x.device)
            clusters = []
            while not assigned.all():
                seed_idx = torch.argmax(s_b.masked_fill(assigned, -1e9))
                cluster = [seed_idx.item()]
                assigned[seed_idx] = True

                # 向后扩展
                for t in range(seed_idx + 1, min(T, seed_idx + self.max_span + 1)):
                    if assigned[t]: break
                    if len(cluster) >= self.max_span: break
                    # score = sim_b[seed_idx, t] - self.beta * s_b[t]
                    score = sim_score[seed_idx, t]
                    if score > dynamic_threshold:
                        cluster.append(t)
                        assigned[t] = True
                    else:
                        break

                # 向前扩展
                for t in range(seed_idx - 1, max(-1, seed_idx - self.max_span - 1), -1):
                    if assigned[t]: break
                    if len(cluster) >= self.max_span: break
                    # score = sim_b[seed_idx, t] - self.beta * s_b[t]
                    score = sim_score[seed_idx, t]
                    if score > dynamic_threshold:
                        cluster.insert(0, t)
                        assigned[t] = True
                    else:
                        break

                clusters.append(cluster)

            ##### !!!!!! clusters没有在时序上排序 [37, 38, 39], [92, 93, 94, 95, 96, 97], [48, 49, 50] 输出的cluster是这样的丧失了时序！！！
            clusters.sort(key=lambda x: x[0])
            # pdb.set_trace()

            # # 聚类中心（均值meanpooling）
            # cluster_embs = []
            # cluster_lengths = []
            # for c in clusters:
            #     # pdb.set_trace()
            #     pooled = x_b[c].mean(dim=0)  # [D]
            #     cluster_embs.append(pooled)
            #     cluster_lengths.append(len(c))
            # cluster_embs = torch.stack(cluster_embs, dim=0).transpose(0, 1)  # [D, N_cluster]
            # 向量化 mean pooling
            N_cluster = len(clusters)
            cluster_mask = torch.zeros(N_cluster, T, dtype=torch.float32, device=x.device)
            for i, c in enumerate(clusters):
                cluster_mask[i, c] = 1.0
            cluster_lengths = cluster_mask.sum(dim=1).long().tolist()
            cluster_mask = cluster_mask / cluster_mask.sum(dim=1, keepdim=True) #  用Meanpooling，注释掉就是改为Sumpooling
            cluster_embs = cluster_mask @ x_b  # [N_cluster, D]
            cluster_embs = cluster_embs.transpose(0, 1)  # [D, N_cluster]

            cluster_embs_list.append(cluster_embs)
            cluster_lengths_list.append(cluster_lengths)

        # padding 输出
        max_len = max(c.size(1) for c in cluster_embs_list)
        cluster_embs_padded = torch.stack([
            F.pad(c, (0, max_len - c.size(1))) for c in cluster_embs_list
        ], dim=0)  # [B, D, max_N_cluster]
        # pdb.set_trace()

        cluster_lengths_tensor = torch.zeros(len(cluster_lengths_list), max_len, dtype=torch.long, device=x.device)
        for b, length_list in enumerate(cluster_lengths_list):
            cluster_lengths_tensor[b, :len(length_list)] = torch.tensor(length_list, dtype=torch.long, device=x.device) # [B, max_N_cluster]

        return cluster_embs_padded, cluster_lengths_tensor

    # def _pairwise_similarity(self, x):  # x: [B, T, D]
    #     dist = torch.cdist(x, x, p=2)  # [B, T, T]
    #     sim = torch.exp(-dist ** 2 * math.log(2))  # φ(xi, xj)
    #     return sim
    def _pairwise_similarity(self, x):  # x: [B, T, D]
        x = F.normalize(x, dim=-1)  # 单位化向量
        sim = torch.matmul(x, x.transpose(1, 2))  # [B, T, T]
        sim = (sim + 1) / 2
        return sim


    def _local_density(self, sim):  # sim: [B, T, T]
        knn_vals = sim.topk(self.k + 1, dim=-1).values[:, :, 1:]  # 去掉自己
        rho = torch.exp(-knn_vals.mean(dim=-1))  # [B, T]
        return rho

    # def _delta(self, sim, rho):  # sim: [B, T, T], rho: [B, T]
    #     B, T = rho.size()
    #     delta = torch.zeros_like(rho)
    #     for b in range(B):
    #         for i in range(T):
    #             higher = torch.where(rho[b] > rho[b, i])[0]
    #             if len(higher) > 0:
    #                 delta[b, i] = sim[b, i, higher].min()
    #             else:
    #                 delta[b, i] = sim[b, i].max()
    #     return delta
    def _delta(self, sim, rho):  # sim: [B, T, T], rho: [B, T]
        B, T = rho.shape

        # 创建一个 mask，表示哪些位置的点密度比当前点高
        # mask[b, i, j] = True 表示 rho[b, j] > rho[b, i]
        rho_i = rho.unsqueeze(2)  # [B, T, 1]
        rho_j = rho.unsqueeze(1)  # [B, 1, T]
        higher_mask = rho_j > rho_i  # [B, T, T]

        # 将不满足条件的位置设为 inf，便于 min 操作
        sim_masked = sim.masked_fill(~higher_mask, float('inf'))  # [B, T, T]
        delta_min, _ = sim_masked.min(dim=2)  # [B, T]

        # 某些点没有比它密度更高的点，这时我们取 sim 最大值作为 delta
        no_higher = ~higher_mask.any(dim=2)  # [B, T]
        delta_max, _ = sim.max(dim=2)  # [B, T]

        # 将这些位置替换为最大 sim
        delta_min[no_higher] = delta_max[no_higher]

        return delta_min  # [B, T]

