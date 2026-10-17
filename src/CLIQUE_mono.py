"""
CLIQUE_mono.py  —  带单调性约束的 CLIQUE 模型
SEMO
继承 CLIQUEestimator，只重写 build_model() 方法：
  - chain-aware batching：同一条链的模式在同一个 batch 中
  - 双 loss：MSLE（有标签样本）+ hinge monotonicity loss（链内相邻对）

放置位置: src/CLIQUE_mono.py
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.util as ut
from src.CLIQUE import (
    CLIQUEestimator,
    CLIQUEdataset,
    collate_fn_hybrid,
    evaluate_hybrid_model,
    get_hybrid_dl_valid,
    get_hybrid_dl_test,
)
from src.model import msle_loss
from src.util import hinge_monotonicity_loss


class CLIQUEestimator_mono(CLIQUEestimator):
    """
    带单调性约束的 CLIQUE 模型。

    新增属性:
        lambda_mono: float, 单调性 loss 的权重
        chain_index: list of dict, 链结构索引（从 chain_index.json 加载）
    """

    def __init__(self):
        super().__init__()
        self.lambda_mono = 1.0
        self.mono_tolerance = 0.0
        self.chain_index = None

    def _build_chain_pattern_to_bound(self, all_chain_patterns):
        """
        为链内所有模式查询 N-gram 表上界。

        参数:
            all_chain_patterns: set of str, 所有链内出现的模式

        返回:
            dict: {pattern: max_bound}
        """
        pattern_bound = {}
        for pat in all_chain_patterns:
            if pat == "%":
                pattern_bound[pat] = self.n_db
            else:
                bound = self.get_bound(pat)
                pattern_bound[pat] = bound
        return pattern_bound

    def _prepare_chain_batches(self, train_data, chain_index, char_dict, bs):
        """
        构建 chain-aware batches。

        每个 batch 包含若干条完整链的所有模式。
        有标签的模式参与 MSLE loss，所有相邻对参与单调性 loss。

        参数:
            train_data: list of (query, card), 有标签的训练数据
            chain_index: list of dict, 链结构
            char_dict: dict, 字符编码字典
            bs: int, batch size（以模式数计）

        返回:
            list of dict: 每个 batch 包含:
                - "queries": list of str
                - "cards": list of float (-1 表示无标签)
                - "bounds": list of float
                - "pair_tight_idx": list of int (链内相邻对的精确端在 batch 内的下标)
                - "pair_loose_idx": list of int (链内相邻对的宽松端在 batch 内的下标)
        """
        # 建立 pattern -> card 的映射
        card_dict = {}
        for query, card in train_data:
            card_dict[query] = card
        # '%' 的基数
        card_dict["%"] = self.n_db

        # 收集所有链内模式，查上界
        all_chain_patterns = set()
        for chain_info in chain_index:
            for pat in chain_info["patterns"]:
                all_chain_patterns.add(pat)
        pattern_bound = self._build_chain_pattern_to_bound(all_chain_patterns)

        # 打乱链顺序
        chain_order = list(range(len(chain_index)))
        np.random.shuffle(chain_order)

        # 组装 batches
        batches = []
        current_batch = {
            "queries": [], "cards": [], "bounds": [],
            "pair_tight_idx": [], "pair_loose_idx": [],
        }

        for ci in chain_order:
            chain_info = chain_index[ci]
            patterns = chain_info["patterns"]
            is_anchor = chain_info["is_anchor"]

            # 记录这条链在当前 batch 中的起始位置
            offset = len(current_batch["queries"])

            for i, (pat, anchor) in enumerate(zip(patterns, is_anchor)):
                # card
                if pat in card_dict:
                    card = card_dict[pat]
                elif pat == "%":
                    card = self.n_db
                else:
                    card = -1  # 无标签

                # bound
                bound = pattern_bound.get(pat, self.n_db)

                current_batch["queries"].append(pat)
                current_batch["cards"].append(float(card))
                current_batch["bounds"].append(float(bound))

            # 链内相邻对
            for i in range(len(patterns) - 1):
                tight_idx = offset + i
                loose_idx = offset + i + 1
                current_batch["pair_tight_idx"].append(tight_idx)
                current_batch["pair_loose_idx"].append(loose_idx)

            # 如果 batch 满了，保存并新建
            if len(current_batch["queries"]) >= bs:
                batches.append(current_batch)
                current_batch = {
                    "queries": [], "cards": [], "bounds": [],
                    "pair_tight_idx": [], "pair_loose_idx": [],
                }

        # 最后一个不完整的 batch
        if current_batch["queries"]:
            batches.append(current_batch)

        return batches

    def _encode_batch(self, batch_dict, char_dict):
        """
        将 batch_dict 编码为模型可接受的 tensor 格式。

        返回:
            xs: tuple (encoded_seqs, lens, max_bounds)
            ys: Tensor of cards
            pair_tight_idx: list of int
            pair_loose_idx: list of int
        """
        patterns = [
            ut.canonicalize_like_query(q, is_last_flip=True)
            for q in batch_dict["queries"]
        ]
        encoded = ut.string_encoding(patterns, char_dict)
        lens = [len(e) for e in encoded]

        # Pad sequences
        padded = ut.keras_pad_sequences(encoded, padding="post")

        xs_tensor = torch.LongTensor(padded)
        bounds_tensor = torch.FloatTensor(batch_dict["bounds"])
        ys_tensor = torch.FloatTensor(batch_dict["cards"])

        xs = (xs_tensor, lens, bounds_tensor)
        return xs, ys_tensor, batch_dict["pair_tight_idx"], batch_dict["pair_loose_idx"]

    def build_model(self, train_data, valid_data, test_data):
        """
        重写训练循环：chain-aware batching + MSLE + 单调性 loss。
        """
        model = self
        lr = self.lr
        l2 = self.l2
        n_epoch = self.n_epoch
        bs = self.bs
        patience = self.patience
        sw = self.sw
        seed = self.seed
        char_dict = self.char_dict
        model_path = self.model_path
        packed = self.packed
        lambda_mono = self.lambda_mono
        mono_tolerance = getattr(self, "mono_tolerance", 0.0)
        chain_index = self.chain_index

        # 如果没有链索引或 lambda=0，退回原始训练
        if chain_index is None or lambda_mono == 0:
            labeled_train_data = ut.labeled_training_rows(train_data)
            print(
                "No chain index or lambda_mono=0, falling back to standard "
                f"training with {len(labeled_train_data)}/{len(train_data)} "
                "labeled rows"
            )
            return super().build_model(
                labeled_train_data, valid_data, test_data
            )

        # 有标签数据加上 bound（用于 valid/test 的标准评估）
        train_data_with_bound = self.append_bound(train_data, packed=False)
        valid_data = self.append_bound(valid_data)
        test_data = self.append_bound(test_data)

        # 标准 valid/test dataloader（用于评估，不变）
        dl_valid = get_hybrid_dl_valid(valid_data, bs, char_dict)
        dl_test = get_hybrid_dl_test(test_data, bs, char_dict)

        ut.set_seed(seed)
        model.cuda()
        optim = Adam(model.parameters(), lr=lr, weight_decay=l2)
        bs_step = 0
        best_val_score = float("inf")
        best_epoch = -1
        best_test_score = -1
        epochs_since_improvement = 0

        build_time = 0
        n_labeled = sum(1 for _, card in train_data if card >= 0)
        n_unlabeled = len(train_data) - n_labeled

        print(f"\n{'=' * 60}")
        print(f"Training with monotonicity constraint")
        print(f"  lambda_mono = {lambda_mono}")
        print(f"  mono_tolerance = {mono_tolerance}")
        print(f"  chains = {len(chain_index)}")
        print(f"  train samples = {len(train_data)} total")
        print(f"  labeled train samples = {n_labeled}")
        print(f"  unlabeled train samples = {n_unlabeled}")
        print(f"{'=' * 60}\n")

        for epoch in range(1, n_epoch + 1):
            start_time = time.time()
            model.train()

            # 每个 epoch 重新构建 chain-aware batches（打乱链顺序）
            np.random.seed(seed + epoch)
            chain_batches = self._prepare_chain_batches(
                train_data, chain_index, char_dict, bs
            )

            epoch_msle_loss = 0.0
            epoch_mono_loss = 0.0
            n_batches = 0

            for batch_dict in tqdm(
                chain_batches,
                total=len(chain_batches),
                desc=f"[Epoch {epoch:3d}/{n_epoch:3d}]",
            ):
                optim.zero_grad()

                # 编码
                xs, ys, pair_tight_idx, pair_loose_idx = self._encode_batch(
                    batch_dict, char_dict
                )

                # 移到 GPU
                xs = (xs[0].cuda(), xs[1], xs[2].cuda())
                ys = ys.cuda()

                # 前向传播（所有模式，包括无标签的）
                preds = model(xs)

                # ---- MSLE loss（仅有标签样本）----
                labeled_mask = ys >= 0
                if labeled_mask.any():
                    loss_msle = msle_loss(ys[labeled_mask], preds[labeled_mask])
                else:
                    loss_msle = torch.tensor(0.0, device=preds.device)

                # ---- 单调性 loss（链内相邻对）----
                if pair_tight_idx and pair_loose_idx:
                    tight_idx = torch.LongTensor(pair_tight_idx)
                    loose_idx = torch.LongTensor(pair_loose_idx)
                    preds_tight = preds[tight_idx]
                    preds_loose = preds[loose_idx]
                    loss_mono = hinge_monotonicity_loss(
                        preds_tight,
                        preds_loose,
                        tolerance=mono_tolerance,
                    )
                else:
                    loss_mono = torch.tensor(0.0, device=preds.device)

                # ---- 总 loss ----
                loss = loss_msle + lambda_mono * loss_mono
                loss.backward()
                optim.step()

                epoch_msle_loss += float(loss_msle.detach().cpu())
                epoch_mono_loss += float(loss_mono.detach().cpu())
                n_batches += 1
                bs_step += 1

                # Tensorboard
                sw.add_scalars("Loss", {
                    "train_msle": float(loss_msle.detach().cpu()),
                    "train_mono": float(loss_mono.detach().cpu()),
                    "train_total": float(loss.detach().cpu()),
                }, global_step=bs_step)

            avg_msle = epoch_msle_loss / max(n_batches, 1)
            avg_mono = epoch_mono_loss / max(n_batches, 1)

            # ---- Validation ----
            losses, q_errs, estimates = evaluate_hybrid_model(model, dl_valid)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)

            sw.add_scalars("Loss", {"valid": loss_}, global_step=bs_step)
            sw.add_scalars("ACC", {"valid": avg_q_err_}, global_step=bs_step)
            sw.add_scalars("ACC", {"valid_max": max_q_err_}, global_step=bs_step)

            end_time = time.time()
            build_time += end_time - start_time

            print(f"  Epoch {epoch:3d}: msle={avg_msle:.4f} mono={avg_mono:.4f} "
                  f"val_q_err={avg_q_err_:.4f} val_max={max_q_err_:.2f}")

            q_err_ = avg_q_err_
            if q_err_ < best_val_score:
                best_val_score = q_err_
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # ---- Test ----
            losses, q_errs, estimates = evaluate_hybrid_model(model, dl_test)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            sw.add_scalars("Loss", {"test": loss_}, global_step=bs_step)
            sw.add_scalars("ACC", {"test": avg_q_err_}, global_step=bs_step)
            sw.add_scalars("ACC", {"test_max": max_q_err_}, global_step=bs_step)

            if epochs_since_improvement == 0:
                best_test_score = avg_q_err_

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered.")
                break

        print(
            f"{best_val_score = }, {best_test_score = }, {patience = }, {best_epoch = }"
        )

        model.load_state_dict(torch.load(model_path))
        return build_time
