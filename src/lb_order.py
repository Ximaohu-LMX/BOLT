"""LB+Order: BOLT's chain learner with no statistical upper-bound access.

The all-percent semantic lower bound remains N, as in the paired BOLT.
All other lower bounds use exactly the inherited witness library/rules.
"""
import torch
from src.CLIQUE_bound_mono import CLIQUEestimator_bound_mono
from src.CLIQUE_bound import get_bound_dl_test, _batch2cuda_bound


class ForbiddenUpperTable:
    def __getitem__(self, key):
        raise RuntimeError('LB+Order must not access an upper-bound table')
    __contains__ = __getitem__
    get = __getitem__


class LBOrderEstimator(CLIQUEestimator_bound_mono):
    def __init__(self):
        super().__init__()
        self.hashtable = ForbiddenUpperTable()
        self.raw_lb_cache = {}

    def create_model(self):
        if self.bound_in:
            raise ValueError('LB+Order requires bound_in=False')
        super().create_model()

    def build_count(self, data_name):
        raise RuntimeError('LB+Order must not build/load upper-bound tables')

    def get_bound(self, query):
        return self.n_db

    def get_lower_bound_for_query(self, query):
        if query not in self.raw_lb_cache:
            self.raw_lb_cache[query] = super().get_lower_bound_for_query(query)
        return self.raw_lb_cache[query]

    def get_effective_lower_bound_for_query(self, query, upper_bound=None,
                                           raw_lower_bound=None):
        raw = self.get_lower_bound_for_query(query) if raw_lower_bound is None else raw_lower_bound
        if not 0 <= raw <= self.n_db:
            raise ValueError('Invalid lower-bound count')
        # Preserve the paired model's semantic all-percent lower bound.
        return self.n_db if query and all(ch == '%' for ch in query) else raw

    def logit(self, x):
        if self.bound_in:
            raise ValueError('Upper-bound input was re-enabled')
        # Never allow an old UB tensor to constrain the output interval.
        return super().logit((x[0], x[1], torch.full_like(x[2], self.n_db), x[3]))

    def build(self, data_name, db, train_data, valid_data, test_data):
        self.n_db = len(db)
        if self.substr_library is None:
            raise RuntimeError('Load the frozen paired lower-bound library first')
        return self.build_model(train_data, valid_data, test_data)

    @torch.no_grad()
    def estimate(self, test_queries, to_cuda=True, is_tqdm=False, batch_size=1, **kwargs):
        self.eval()
        data = self.append_bound([(q, 0) for q in test_queries], is_tqdm=is_tqdm)
        values = []
        for batch in get_bound_dl_test(data, batch_size, self.char_dict):
            x, _ = _batch2cuda_bound(batch, to_cuda=to_cuda)
            values.extend(self(x).cpu().tolist())
        return values


def make_model(cfg, chars, writer=None, model_path=None):
    model = LBOrderEstimator()
    model.set_params(cfg['N'], cfg['PT'], cfg['seed'], chars, cfg['ch_es'], 1,
                     False, cfg['rnn_hs'], cfg['pred_hs'], cfg['n_pred_layer'],
                     cfg['learning_rate'], cfg['l2'], cfg['n_epoch'], cfg['batch_size'],
                     cfg['patience'], True, writer, None, model_path, False, False,
                     cfg['dynamicPT'], cfg['max_entry_ratio'], cfg['batch_norm'])
    model.lambda_mono = cfg['lambda_mono']
    model.mono_tolerance = cfg['mono_tolerance']
    for key, value in cfg.items():
        if key.startswith('lb_'):
            setattr(model, key, value)
    return model
