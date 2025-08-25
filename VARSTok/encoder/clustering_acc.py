import torch, math
import torch.nn as nn
import torch.nn.functional as F

class SequentialDPClustering(nn.Module):
    def __init__(self, k=10, beta=0.2, threshold=0.7,
                 use_dynamic_threshold=False, max_span=4):
        super().__init__()
        self.k = k
        self.beta = beta
        self.threshold = threshold
        self.use_dynamic_threshold = use_dynamic_threshold
        self.max_span = max_span

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x : [B, D, T]
        return:
            cluster_embs_padded   : [B, D, T']
            cluster_lengths_tensor: [B, T']
        """
        B, D, T = x.size()
        x_t = x.transpose(1, 2)                                   # [B, T, D]

        # -------- 1) 无梯度求 φ/ρ/δ 及贪心分配 --------
        with torch.no_grad():
            sim   = self._pairwise_similarity(x_t)                # [B,T,T]
            rho   = self._local_density(sim)
            delta = self._delta(sim, rho)
            s     = rho * delta                                   # [B,T]

            all_ids, all_lens = [], []
            for b in range(B):
                sim_b, s_b = sim[b], s[b]

                # 动态阈值
                dyn_th = self.threshold
                if self.use_dynamic_threshold:
                    mu, sg = sim_b.mean().item(), sim_b.std().item()
                    dyn_th = mu + 0.5 * sg

                sim_score = sim_b - self.beta * s_b.view(1, -1)

                assigned = torch.zeros(T, dtype=torch.bool, device=x.device)
                clusters = []
                remaining = T

                while remaining:
                    seed = torch.argmax(s_b.masked_fill(assigned, -1e9))
                    clst = [seed.item()]
                    assigned[seed] = True; remaining -= 1

                    # → 右扩
                    for t in range(seed + 1, min(T, seed + self.max_span + 1)):
                        if assigned[t] or len(clst) >= self.max_span: break
                        if sim_score[seed, t] > dyn_th:
                            clst.append(t); assigned[t] = True; remaining -= 1
                        else: break
                    # ← 左扩
                    for t in range(seed - 1, max(-1, seed - self.max_span - 1), -1):
                        if assigned[t] or len(clst) >= self.max_span: break
                        if sim_score[seed, t] > dyn_th:
                            clst.insert(0, t); assigned[t] = True; remaining -= 1
                        else: break
                    clusters.append(clst)

                clusters.sort(key=lambda c: c[0])                 # 保序

                # ids: [T]，-1 位置不会被用到
                ids = torch.full((T,), -1, dtype=torch.long, device=x.device)
                for cid, c in enumerate(clusters):
                    ids[c] = cid
                all_ids.append(ids)
                all_lens.append([len(c) for c in clusters])

        # -------- 2) 有梯度 —— mean-pool --------
        cluster_embs, max_len = [], max(len(l) for l in all_lens)

        for b in range(B):
            ids = all_ids[b]                                      # [T]
            lens = all_lens[b]                                    # list[int]
            N = len(lens)

            if N == 0:
                cluster_embs.append(torch.zeros(D, 1, device=x.device))
                continue

            # sum-pool  (PyTorch 2.0 用 index_add_)
            sums = torch.zeros(N, D, device=x.device)
            sums.index_add_(0, ids, x_t[b])                       # [N,D]

            lens_t = torch.tensor(lens, device=x.device).unsqueeze(1)
            cluster_embs.append((sums / lens_t).t())              # [D,N]

        cluster_embs_padded = torch.stack(
            [F.pad(c, (0, max_len - c.size(1))) for c in cluster_embs], dim=0
        )                                                         # [B,D,max_len]

        cluster_lengths_tensor = torch.zeros(B, max_len,
                                             dtype=torch.long,
                                             device=x.device)
        for b, l in enumerate(all_lens):
            cluster_lengths_tensor[b, :len(l)] = torch.tensor(l, device=x.device)

        return cluster_embs_padded, cluster_lengths_tensor

    def _pairwise_similarity(self, x):
        x = F.normalize(x, dim=-1)
        return (torch.matmul(x, x.transpose(1, 2)) + 1) / 2

    def _local_density(self, sim):
        knn = sim.topk(self.k + 1, dim=-1).values[:, :, 1:]
        return torch.exp(-knn.mean(dim=-1))

    def _delta(self, sim, rho):
        rho_i, rho_j = rho.unsqueeze(2), rho.unsqueeze(1)
        mask = rho_j > rho_i
        sim_masked = sim.masked_fill(~mask, float('inf'))
        delta_min, _ = sim_masked.min(dim=2)
        no_higher = ~mask.any(dim=2)
        delta_max, _ = sim.max(dim=2)
        delta_min[no_higher] = delta_max[no_higher]
        return delta_min