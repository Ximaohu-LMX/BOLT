"""
UB_LB.py  —  带上下界双边约束的 CLIQUE 模型

核心改动：
  原始 CLIQUE:  pred = upper_bound * sigmoid(logit)        范围 [0, UB]
  本文件:       pred = LB + (UB - LB) * sigmoid(logit)     范围 [LB, UB]

  当 LB > 0 时，模型的搜索空间被压缩，低估现象减少。
  当 LB = 0 时，退化为原始 CLIQUE 行为。

继承关系: UBLBEstimator -> CLIQUEestimator
  重写: logit(), append_bound()
  新增: build_lower_bound_table(), get_lower_bound()
  修改: UBLBDataset, collate_fn_hybrid_bound（增加 lower_bound 字段）

放置位置: src/UB_LB.py

用法:
  # 在 main 脚本中:
  from src.UB_LB import UBLBEstimator
  model = UBLBEstimator()
  model.set_params(...)
  model.build(data_name, db, train_data, valid_data, test_data)
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

import src.util as ut
from src.CLIQUE import (
    CLIQUEestimator,
    CLIQUEdataset,
    collate_fn_hybrid,
    evaluate_hybrid_model,
)
from src.lower_bound import (
    build_freq_string_library,
    get_lower_bound,
    get_cached_freq_library,
    save_freq_library,
)


# =====================================================
#  Dataset 和 Collate（增加 lower_bound）
# =====================================================

class UBLBDataset(Dataset):
    """
    扩展 CLIQUEdataset，增加 lower_bound 字段。

    data 格式: [(query, card, upper_bound, lower_bound), ...]
    """

    def __init__(self, data, char_dict):
        super().__init__()
        self.data = data
        self.patterns = [
            ut.canonicalize_like_query(x[0], is_last_flip=True) for x in data
        ]
        self.lens = [len(x) for x in self.patterns]
        self.ys = [x[1] for x in data]
        self.upper_bounds = [x[2] for x in data]
        self.lower_bounds = [x[3] for x in data]
        self.encoded_sequences = ut.string_encoding(self.patterns, char_dict)
        self.char_dict = char_dict

    def __getitem__(self, index):
        return (
            self.encoded_sequences[index],
            self.lens[index],
            self.upper_bounds[index],
            self.lower_bounds[index],
            self.ys[index],
        )

    def __len__(self):
        return len(self.data)


def collate_fn_hybrid_bound(batch):
    """Collate 函数，处理 (xs, lens, upper_bounds, lower_bounds, ys)。"""
    batch = tuple(zip(*batch))
    xs, lens, upper_bounds, lower_bounds, ys = batch

    if len(xs) > 1:
        xs = ut.keras_pad_sequences(xs, padding="post")

    batch = (
        torch.LongTensor(xs),
        lens,
        torch.FloatTensor(upper_bounds),
        torch.FloatTensor(lower_bounds),
        torch.FloatTensor(ys),
    )
    return batch


def get_bound_dl_train(train_data, bs, char_dict, shuffle=True):
    ds = UBLBDataset(train_data, char_dict)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      collate_fn=collate_fn_hybrid_bound)


def get_bound_dl_test(test_data, bs, char_dict):
    return get_bound_dl_train(test_data, bs, char_dict, shuffle=False)


def evaluate_bound_model(model, dl_test):
    """评估函数，适配 5 元组 batch。"""
    losses = []
    q_errs = []
    estimates = []
    model.eval()
    to_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for batch in dl_test:
            xs, ys = _batch2cuda_bound(batch, to_cuda=to_cuda)
            loss = model.loss(xs, ys, reduction="none")
            loss_ = loss.cpu().numpy()
            losses.extend(loss_)
            preds = model(xs)
            preds_ = preds.cpu().numpy()
            estimates.extend(preds_)
            q_err = ut.mean_Q_error(ys.cpu(), preds_, reduction="none")
            q_errs.extend(q_err)
    return losses, q_errs, estimates


def _batch2cuda_bound(batch, to_cuda=True):
    """将 5 元组 batch 拆分为 xs 和 ys。"""
    # batch = (xs_tensor, lens, upper_bounds, lower_bounds, ys)
    xs = []
    ys = batch[-1]
    for i in range(len(batch) - 1):
        elem = batch[i]
        if isinstance(elem, torch.Tensor) and to_cuda:
            elem = elem.cuda()
        xs.append(elem)
    xs = tuple(xs)
    if isinstance(ys, torch.Tensor) and to_cuda:
        ys = ys.cuda()
    return xs, ys


# =====================================================
#  UBLBEstimator
# =====================================================

class UBLBEstimator(CLIQUEestimator):
    """
    带上下界双边约束的 CLIQUE 模型。

    pred = lower_bound + (upper_bound - lower_bound) * sigmoid(logit)

    新增属性:
        substr_library: dict, 高频字符串库
        use_lower_bound: bool, 是否启用下界（默认 True）
    """

    def __init__(self):
        super().__init__()
        self.substr_library = None
        self.use_lower_bound = True

    def build_lower_bound_table(self, data_name, db=None,
                                 top_k_strings=500, top_k_substrings=500,
                                 min_substr_len=3, max_substr_len=15):
        """
        构建高频字符串库。

        参数:
            data_name: str
            db: list of str（可选，不传则从文件加载）
        """
        cache_path = f"res/{data_name}/freq_library/lib_s{top_k_strings}_sub{top_k_substrings}.pkl"

        if os.path.exists(cache_path):
            from src.lower_bound import load_freq_library
            self.substr_library = load_freq_library(cache_path)
            print(f"Loaded cached frequency library from {cache_path}")
        else:
            if db is None:
                db_path = f"data/{data_name}/{data_name}.txt"
                db = ut.read_strings(db_path)

            self.substr_library = build_freq_string_library(
                db,
                top_k_strings=top_k_strings,
                top_k_substrings=top_k_substrings,
                min_substr_len=min_substr_len,
                max_substr_len=max_substr_len,
            )
            save_freq_library(self.substr_library, cache_path)

        return self.substr_library

    def get_lower_bound_for_query(self, query):
        """计算单个查询的下界。"""
        if self.substr_library is None or not self.use_lower_bound:
            return 0
        return get_lower_bound(query, self.substr_library)

    def append_bound(self, labeled_data, packed=False, is_tqdm=True):
        """
        重写 append_bound，增加 lower_bound。

        输入: [(query, card), ...]
        输出: [(query, card, upper_bound, lower_bound), ...]
        """
        n_db = self.n_db

        output = []
        if is_tqdm:
            labeled_data_p = tqdm(labeled_data)
            labeled_data_p.set_description("[Append Bound (UB+LB)]")
        else:
            labeled_data_p = labeled_data

        for query, card in labeled_data_p:
            upper_bound = self.get_bound(query)  # N-gram 表上界
            lower_bound = self.get_lower_bound_for_query(query)

            # 确保 LB <= UB，且 LB <= card（安全检查）
            lower_bound = min(lower_bound, upper_bound)

            output.append([query, card, upper_bound, lower_bound])

        # 统计下界覆盖率
        n_with_lb = sum(1 for x in output if x[3] > 0)
        print(f"  Lower bound coverage: {n_with_lb}/{len(output)} "
              f"({100*n_with_lb/max(len(output),1):.1f}%) queries have LB > 0")

        return output

    def logit(self, x):
        """
        重写 logit，使用双边约束：
          pred = lower_bound + (upper_bound - lower_bound) * sigmoid(logit)
        """
        # x = (xs, lens, upper_bounds, lower_bounds)
        xs, lens, upper_bounds, lower_bounds = x[0], x[1], x[2], x[3]

        embeds = self.embedding(xs)
        lens_tensor = torch.LongTensor(lens)
        bs = lens_tensor.shape[0]

        packed_input = pack_padded_sequence(
            embeds, lens_tensor, batch_first=True, enforce_sorted=False
        )
        output, hidden_ = self.rnn(packed_input)

        if isinstance(self.rnn, torch.nn.LSTM):
            hidden, cell = hidden_
        else:
            hidden = hidden_
        hidden = hidden.reshape(bs, -1)

        if self.bound_in:
            hidden = torch.concat(
                [hidden, torch.unsqueeze(torch.log(upper_bounds + 1), -1)], dim=-1
            )

        preds = self.pred_layer(hidden)
        preds = torch.squeeze(preds, dim=-1)
        preds = torch.sigmoid(preds)

        # 核心改动: pred = LB + (UB - LB) * sigmoid
        range_bounds = upper_bounds - lower_bounds
        range_bounds = F.relu(range_bounds)  # 确保非负
        preds = lower_bounds + range_bounds * preds

        return preds

    def forward(self, x):
        out = self.logit(x)
        if not self.training:
            out = F.relu(out)
        return out

    def loss(self, x, y, reduction="mean", mask=None):
        """Loss 计算（与原始相同，只是走新的 logit）。"""
        from src.model import msle_loss
        preds = self.forward(x)
        res = msle_loss(y, preds, reduction=reduction)
        return res

    def build_model(self, train_data, valid_data, test_data):
        """重写训练循环，使用带下界的 dataset 和 collate。"""
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

        # append_bound 现在返回 4 元组
        train_data = self.append_bound(train_data)
        valid_data = self.append_bound(valid_data)
        test_data = self.append_bound(test_data)

        dl_train = get_bound_dl_train(train_data, bs, char_dict)
        dl_valid = get_bound_dl_test(valid_data, bs, char_dict)
        dl_test = get_bound_dl_test(test_data, bs, char_dict)

        ut.set_seed(seed)
        model.cuda()
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        bs_step = 0
        best_val_score = float("inf")
        best_epoch = -1
        best_test_score = -1
        epochs_since_improvement = 0
        build_time = 0

        for epoch in range(1, n_epoch + 1):
            start_time = time.time()
            model.train()

            for batch in tqdm(
                dl_train, total=len(dl_train),
                desc=f"[Epoch {epoch:3d}/{n_epoch:3d}]"
            ):
                optim.zero_grad()
                xs, ys = _batch2cuda_bound(batch)
                loss = model.loss(xs, ys)
                loss.backward()
                optim.step()

                loss_ = float(loss.detach().cpu().numpy())
                bs_step += 1
                sw.add_scalars("Loss", {"train": loss_}, global_step=bs_step)

                with torch.no_grad():
                    preds_ = model(xs).cpu().numpy()
                avg_q_err_ = ut.mean_Q_error(ys.cpu(), preds_)
                max_q_err_ = ut.mean_Q_error(ys.cpu(), preds_, reduction="max")
                sw.add_scalars("ACC", {"train": avg_q_err_}, global_step=bs_step)
                sw.add_scalars("ACC", {"train_max": max_q_err_}, global_step=bs_step)

            # Valid
            losses, q_errs, estimates = evaluate_bound_model(model, dl_valid)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            sw.add_scalars("Loss", {"valid": loss_}, global_step=bs_step)
            sw.add_scalars("ACC", {"valid": avg_q_err_}, global_step=bs_step)
            sw.add_scalars("ACC", {"valid_max": max_q_err_}, global_step=bs_step)

            end_time = time.time()
            build_time += end_time - start_time

            q_err_ = avg_q_err_
            if q_err_ < best_val_score:
                best_val_score = q_err_
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            # Test
            losses, q_errs, estimates = evaluate_bound_model(model, dl_test)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            sw.add_scalars("ACC", {"test": avg_q_err_}, global_step=bs_step)
            sw.add_scalars("ACC", {"test_max": max_q_err_}, global_step=bs_step)

            if epochs_since_improvement == 0:
                best_test_score = avg_q_err_

            if epochs_since_improvement >= patience:
                print(f"Early stopping triggered.")
                break

        print(f"{best_val_score = }, {best_test_score = }, {patience = }, {best_epoch = }")
        model.load_state_dict(torch.load(model_path))
        return build_time

    def build(self, data_name, db, train_data, valid_data, test_data):
        """重写 build，增加下界表构建。"""
        seed = self.seed
        self.n_db = len(db)
        np.random.seed(seed)

        build_time_count = self.build_count(data_name)
        print(f"{build_time_count = }")

        # 构建下界表
        if self.use_lower_bound:
            self.build_lower_bound_table(data_name, db)

        build_time_model = self.build_model(train_data, valid_data, test_data)
        build_time = build_time_count + build_time_model
        return build_time

    def estimate(self, test_queries, to_cuda=True, is_tqdm=True, **kwargs):
        """
        重写 estimate，增加下界查询。
        """
        from src.CLIQUE import is_LIKEquery_in_Table
        char_dict = self.char_dict

        test_data = []
        estimates = []

        for q_id, test_query in enumerate(test_queries):
            b_in_table = is_LIKEquery_in_Table(
                test_query, self.hashtable, self.N)
            if b_in_table:
                test_query_norm = ut.flip_last_canonicalized_like_query(test_query)
                ngram = test_query_norm.strip("%").replace("_", ut.esc_ac)
                is_prefix_query = test_query_norm[0] != "%"
                is_suffix_query = test_query_norm[-1] != "%"
                if is_prefix_query:
                    ngram = ut.esc_sc + ngram
                if is_suffix_query:
                    ngram = ngram + ut.esc_ec
                card = self.hashtable[ngram]
                estimates.append((card, q_id))
            else:
                test_data.append((test_query, q_id))

        if len(test_data) > 0:
            # append_bound 需要 (query, card) 格式
            test_data_for_bound = [(q, qid) for q, qid in test_data]
            test_data_bounded = []
            for query, q_id in test_data_for_bound:
                ub = self.get_bound(query)
                lb = self.get_lower_bound_for_query(query)
                lb = min(lb, ub)
                test_data_bounded.append((query, q_id, ub, lb))

            ds_test = UBLBDataset(
                [(q, qid, ub, lb) for q, qid, ub, lb in test_data_bounded],
                char_dict,
            )
            dl_test = DataLoader(
                ds_test, batch_size=1, shuffle=False,
                collate_fn=collate_fn_hybrid_bound,
            )

            model = self
            for batch in dl_test:
                if to_cuda:
                    xs, ys = _batch2cuda_bound(batch, to_cuda=to_cuda)
                else:
                    xs, ys = (batch[0], batch[1], batch[2], batch[3]), batch[4]
                preds = model(xs)
                preds_ = preds.detach().cpu().numpy()
                estimates.extend(zip(preds_, ys.tolist()))

        estimates = [x[0] for x in sorted(estimates, key=lambda x: x[1])]
        return estimates