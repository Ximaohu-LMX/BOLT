import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import (
    BatchNorm1d,
    Embedding,
    LeakyReLU,
    Linear,
    LSTM,
    Sequential,
)
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import src.util as ut
from src.CLIQUE import CLIQUEestimator
from src.model import init_weights


class CLIQUEdataset_no_bound(Dataset):
    """LIKE-pattern dataset without upper/lower bound fields."""

    def __init__(self, data, char_dict):
        super().__init__()
        self.data = data
        self.patterns = [
            ut.canonicalize_like_query(x[0], is_last_flip=True) for x in data
        ]
        self.lens = [len(x) for x in self.patterns]
        self.ys = [x[1] for x in data]
        self.encoded_sequences = ut.string_encoding(self.patterns, char_dict)

    def __getitem__(self, index):
        return (
            self.encoded_sequences[index],
            self.lens[index],
            self.ys[index],
        )

    def __len__(self):
        return len(self.data)


def collate_fn_no_bound(batch):
    batch = tuple(zip(*batch))
    xs, lens, ys = batch

    if len(xs) > 1:
        max_len = max(len(x) for x in xs)
        xs = [list(x) + [0] * (max_len - len(x)) for x in xs]

    return torch.LongTensor(xs), lens, torch.FloatTensor(ys)


def get_no_bound_dl(data, bs, char_dict, shuffle):
    ds = CLIQUEdataset_no_bound(data, char_dict)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      collate_fn=collate_fn_no_bound)


def evaluate_no_bound_model(model, dl_test):
    losses = []
    q_errs = []
    estimates = []
    model.eval()
    to_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for batch in dl_test:
            xs, ys = ut.batch2cuda(batch, to_cuda=to_cuda)
            loss = model.loss(xs, ys, reduction="none")
            loss_ = loss.cpu().numpy()
            losses.extend(loss_)
            preds = model(xs)
            preds_ = preds.cpu().numpy()
            estimates.extend(preds_)
            q_err = ut.mean_Q_error(ys.cpu(), preds_, reduction="none")
            q_errs.extend(q_err)
    return losses, q_errs, estimates


class CLIQUEestimator_no_bound(CLIQUEestimator):
    """
    CLIQUE LSTM encoder with a pure log-count regression head.

    No N-gram table, no UB/LB input, and no table-hit shortcut are used. The
    linear head emits z, the loss fits z to log(card + 1), and inference maps z
    back to cardinality space via exp(z) - 1.
    """

    def set_params(
        self,
        N,
        PT,
        seed,
        char_dict,
        ch_es,
        n_rnn,
        packed,
        rnn_hs,
        pred_hs,
        n_pred_layer,
        lr,
        l2,
        n_epoch,
        bs,
        patience,
        last_flip,
        sw,
        count_path,
        model_path,
        bound_in,
        pack_all,
        dynamicPT,
        max_entry_ratio,
        batch_norm,
    ):
        super().set_params(
            N,
            PT,
            seed,
            char_dict,
            ch_es,
            n_rnn,
            False,
            rnn_hs,
            pred_hs,
            n_pred_layer,
            lr,
            l2,
            n_epoch,
            bs,
            patience,
            last_flip,
            sw,
            None,
            model_path,
            False,
            False,
            dynamicPT,
            max_entry_ratio,
            batch_norm,
        )

    def create_model(self):
        ut.set_seed(self.seed)
        self.embedding = Embedding(self.n_char, self.ch_es, padding_idx=0)
        self.rnn = LSTM(
            self.ch_es, self.rnn_hs, batch_first=True, num_layers=self.n_rnn
        )

        self.pred_layer = Sequential()
        pred_hs_list = [self.rnn_hs]
        pred_hs_list.extend([self.pred_hs] * (self.n_pred_layer - 1))
        pred_hs_list.append(1)

        for i, (in_size, out_size) in enumerate(
            zip(pred_hs_list[:-1], pred_hs_list[1:])
        ):
            if self.batch_norm and i == 0:
                self.pred_layer.add_module(f"BN-{i}", BatchNorm1d(in_size))
            if i > 0:
                self.pred_layer.add_module(f"LeakyReLU-{i}", LeakyReLU())
            self.pred_layer.add_module(f"PRED-{i}", Linear(in_size, out_size))
        self.pred_layer.apply(init_weights)

    def logit(self, x):
        xs, lens = x
        embeds = self.embedding(xs)

        lens = torch.LongTensor(lens)
        bs = lens.shape[0]
        packed = pack_padded_sequence(
            embeds, lens, batch_first=True, enforce_sorted=False
        )

        _, hidden_ = self.rnn(packed)
        hidden, _ = hidden_
        hidden = hidden.reshape(bs, -1)

        z = self.pred_layer(hidden)
        return torch.squeeze(z, dim=-1)

    def forward(self, x):
        z = self.logit(x)
        return torch.clamp_min(torch.exp(z) - 1.0, 1.0)

    def loss(self, x, y, reduction="mean", mask=None):
        if mask is not None:
            raise ValueError("CLIQUEestimator_no_bound does not support masks")
        z = self.logit(x)
        target = torch.log(torch.clamp_min(y, 0.0) + 1.0)
        return F.mse_loss(z, target, reduction=reduction)

    def build_count(self, data_name):
        self.hashtable = None
        return 0.0

    def build_model(self, train_data, valid_data, test_data):
        model = self
        ut.assert_no_unlabeled_training_targets(train_data, "NO_BOUND training")
        dl_train = get_no_bound_dl(train_data, self.bs, self.char_dict, True)
        dl_valid = get_no_bound_dl(valid_data, self.bs, self.char_dict, False)
        dl_test = get_no_bound_dl(test_data, self.bs, self.char_dict, False)

        ut.set_seed(self.seed)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            model.cuda()
        optim = Adam(model.parameters(), lr=self.lr, weight_decay=self.l2)
        bs_step = 0
        best_val_score = float("inf")
        best_epoch = -1
        best_test_score = -1
        epochs_since_improvement = 0
        build_time = 0

        for epoch in range(1, self.n_epoch + 1):
            start_time = time.time()
            model.train()

            for batch in tqdm(
                dl_train, total=len(dl_train), desc=f"[Epoch {epoch:3d}/{self.n_epoch:3d}]"
            ):
                optim.zero_grad()
                xs, ys = ut.batch2cuda(batch, to_cuda=use_cuda)
                loss = model.loss(xs, ys)
                loss.backward()
                loss_ = float(loss.detach().cpu().numpy())
                optim.step()

                with torch.no_grad():
                    preds_ = model(xs).cpu().numpy()
                avg_q_err_ = ut.mean_Q_error(ys.cpu(), preds_)
                max_q_err_ = ut.mean_Q_error(ys.cpu(), preds_, reduction="max")
                bs_step += 1
                self.sw.add_scalars("Loss", {"train": loss_}, global_step=bs_step)
                self.sw.add_scalars("ACC", {"train": avg_q_err_}, global_step=bs_step)
                self.sw.add_scalars("ACC", {"train_max": max_q_err_}, global_step=bs_step)

            losses, q_errs, estimates = evaluate_no_bound_model(model, dl_valid)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            self.sw.add_scalars("Loss", {"valid": loss_}, global_step=bs_step)
            self.sw.add_scalars("ACC", {"valid": avg_q_err_}, global_step=bs_step)
            self.sw.add_scalars("ACC", {"valid_max": max_q_err_}, global_step=bs_step)

            end_time = time.time()
            build_time += end_time - start_time

            if avg_q_err_ < best_val_score:
                best_val_score = avg_q_err_
                best_epoch = epoch
                torch.save(model.state_dict(), self.model_path)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            losses, q_errs, estimates = evaluate_no_bound_model(model, dl_test)
            loss_ = sum(losses) / len(losses)
            avg_q_err_ = sum(q_errs) / len(q_errs)
            max_q_err_ = max(q_errs)
            self.sw.add_scalars("Loss", {"test": loss_}, global_step=bs_step)
            self.sw.add_scalars("ACC", {"test": avg_q_err_}, global_step=bs_step)
            self.sw.add_scalars("ACC", {"test_max": max_q_err_}, global_step=bs_step)

            if epochs_since_improvement == 0:
                best_test_score = avg_q_err_

            if epochs_since_improvement >= self.patience:
                print("Early stopping triggered.")
                break

        print(
            f"{best_val_score = }, {best_test_score = }, {self.patience = }, {best_epoch = }"
        )
        map_location = None if torch.cuda.is_available() else "cpu"
        model.load_state_dict(torch.load(self.model_path, map_location=map_location))
        return build_time

    def build(self, data_name, db, train_data, valid_data, test_data):
        self.n_db = len(db)
        np.random.seed(self.seed)
        return self.build_model(train_data, valid_data, test_data)

    def estimate(self, test_queries, to_cuda=True, is_tqdm=True, **kwargs):
        char_dict = self.char_dict
        test_data = [(test_query, q_id) for q_id, test_query in enumerate(test_queries)]
        ds_test = CLIQUEdataset_no_bound(test_data, char_dict)
        dl_test = DataLoader(
            ds_test, batch_size=1, shuffle=False, collate_fn=collate_fn_no_bound
        )

        estimates = []
        model = self
        for batch in dl_test:
            if to_cuda:
                xs, ys = ut.batch2cuda(batch, to_cuda=to_cuda)
            else:
                xs, ys = batch[:-1], batch[-1]
            preds = model(xs)
            preds_ = preds.detach().cpu().numpy()
            estimates.extend(zip(preds_, ys.tolist()))

        return [x[0] for x in sorted(estimates, key=lambda x: x[1])]

    def model_size(self):
        model_size = os.path.getsize(self.model_path)
        print(f"{model_size = }")
        return model_size
