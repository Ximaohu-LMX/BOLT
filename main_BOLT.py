"""Train BOLT (UB+LB+Order).

Use scripts/run_accuracy.sh --dataset AUTHOR --budget A0 --method BOLT
for the paper's complete frozen configuration.
"""

from src.util import read_strings
import src.util as ut
import os
import yaml
import time
import torch
import numpy as np
from tqdm import tqdm

from src.BOLT import BOLTEstimator
from src.phase_timing import timer as efficiency_timer
from src.util import compute_chain_mono_score, compute_mono_score, load_chain_index

MODE_EST = str(os.getenv("MODE")).lower() == "est"
MODE_SIZE = str(os.getenv("MODE")).lower() == "size"
SKIP_LEGACY_ORDER_EVAL = str(os.getenv("SKIP_LEGACY_ORDER_EVAL", "")).lower() in {
    "1", "true", "yes", "y", "on",
}


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


BOLT_PROFILE_ENABLED = _env_flag("BOLT_PROFILE")
bolt_profile_startup_timings = {}

input_config_path = "configs/CLIQUE/default.yml"
default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
parser.add_argument("--lambda_order", type=float, default=1.0,
                    help="Weight for monotonicity loss")
parser.add_argument("--order_tolerance", type=float, default=0.0,
                    help="Relative tolerance for monotonicity loss dead-zone")
parser.add_argument("--chain_index", type=str, default=None,
                    help="Path to chain_index.json")
parser.add_argument("--chain_sample_ratio", type=float, default=1.0,
                    help="Ratio of whole chains kept with monotonic edges; labeled rows from unselected chains are retained")
parser.add_argument(
    "--force-chain-training", "--force_chain_training",
    dest="force_chain_training", action="store_true",
    help="Keep chain-aware batches and forward path when lambda_order is zero",
)
parser.add_argument("--lb_min_substr_len", type=int, default=5,
                    help="Minimum frequent substring length for lower bounds")
parser.add_argument("--lb_max_substr_len", type=int, default=10,
                    help="Maximum frequent substring length for lower bounds")
parser.add_argument("--lb_top_k_strings", type=int, default=500,
                    help="Number of frequent complete strings for lower bounds")
parser.add_argument("--lb_top_k_substrings", type=int, default=500,
                    help="Number of frequent substrings per category for lower bounds")
parser.add_argument("--lb_min_freq_ratio", type=float, default=0.0005,
                    help="Minimum frequency ratio for frequent substrings")
args = parser.parse_args()
args.chain_sampling_policy = "whole_chain_edges_preserve_labeled_rows_v1"
if args.force_chain_training and args.lambda_order != 0:
    parser.error("--force-chain-training is only valid with --lambda_order 0")

# ---- 参数处理（与 main_CLIQUE.py 一致）----
max_entry_ratio_dict = {
    "WIKI": 4.35, "IMDB": 1.675, "DBLP": 3.12, "GENE": 204.8,
    "AUTHOR": 4.8, "DBLP-AN": 3.15, "IMDb-AN": 2.6, "IMDb-MT": 1.6,
    "TPCH-PN": 15.0, "link_type.link": 8.0, "movie_companies.note": 0.99,
    "name.name": 0.359, "title.title": 0.223,
}
rnn_hs_dict = {
    "DBLP-AN": 160, "IMDb-AN": 160, "IMDb-MT": 160, "TPCH-PN": 160,
    "link_type.link": 181, "movie_companies.note": 154,
    "name.name": 152, "title.title": 151,
}
if args.data_name in max_entry_ratio_dict:
    args.max_entry_ratio = max_entry_ratio_dict[args.data_name]
if args.data_name in rnn_hs_dict:
    args.rnn_hs = rnn_hs_dict[args.data_name]

args.model_name = "CLIQUE"
args = ut.add_just_params(args)
method_name = "BOLT_L0" if args.force_chain_training else "UB_LB" if args.lambda_order == 0 else "BOLT"
args.model_name = args.model_name.replace("CLIQUE", method_name, 1)

if args.packed:
    args.model_name += "-PACK"
if not args.crh:
    args.model_name += "-T"
if args.pack_all:
    args.model_name += "-ALL"
if args.table_only:
    args.model_name = "TABLE"

args.tag = os.path.join(args.tag, args.model_name)
print(args)

# ---- 参数 ----
lambda_order = args.lambda_order
order_tolerance = args.order_tolerance
data_name = args.data_name
workload = args.workload
seed = args.seed

(
    db_path, model_config_path, model_path,
    est_path, est_time_path, stat_path,
) = ut.get_common_pathes_for_estimator(data_name, args.tag, workload, seed)

sw = ut.get_summary_writer(data_name, args.tag, workload, seed)
count_path = ut.get_count_path_for_HybridEstimator(
    data_name, args.PT, args.N, args.dynamicPT, args.max_entry_ratio
)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

# ---- 加载数据 ----
_t0 = time.perf_counter() if BOLT_PROFILE_ENABLED else None
db = read_strings(db_path)
n_db = len(db)
if BOLT_PROFILE_ENABLED:
    bolt_profile_startup_timings["read_db"] = time.perf_counter() - _t0

_t0 = time.perf_counter() if BOLT_PROFILE_ENABLED else None
char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, args.p_train)
if BOLT_PROFILE_ENABLED:
    bolt_profile_startup_timings["get_training_data"] = time.perf_counter() - _t0
efficiency_timer.mark("startup_and_data_load")

# ---- 加载链索引 ----
chain_index_path = args.chain_index
if chain_index_path is None:
    chain_index_path = f"data/{data_name}/training/{workload}/chain_index.json"

chain_index = None
chain_sampling_stats = {}
_t0 = time.perf_counter() if BOLT_PROFILE_ENABLED else None
if os.path.exists(chain_index_path):
    chain_index = load_chain_index(chain_index_path)
    print(f"Loaded chain index: {len(chain_index)} chains from {chain_index_path}")
    chain_index, chain_sampling_stats = ut.sample_chain_index_preserve_labeled(
        chain_index, train_data, args.chain_sample_ratio, seed
    )
    print("Chain sampling:")
    for name, value in chain_sampling_stats.items():
        print(f"  {name}: {value}")
else:
    print(f"WARNING: chain_index.json not found at {chain_index_path}")
    if args.force_chain_training:
        raise FileNotFoundError(
            f"Strict chain training requires chain_index.json: {chain_index_path}"
        )
    print(f"  Falling back to UB_LB (no monotonicity)")
    lambda_order = 0.0
if BOLT_PROFILE_ENABLED:
    bolt_profile_startup_timings["load_chain_index"] = time.perf_counter() - _t0

# ---- 创建模型 ----
_t0 = time.perf_counter() if BOLT_PROFILE_ENABLED else None
model = BOLTEstimator()
model.lb_min_substr_len = args.lb_min_substr_len
model.lb_max_substr_len = args.lb_max_substr_len
model.lb_top_k_strings = args.lb_top_k_strings
model.lb_top_k_substrings = args.lb_top_k_substrings
model.lb_min_freq_ratio = args.lb_min_freq_ratio
model.set_params(
    args.N, args.PT, seed, char_dict, args.ch_es, 1, args.packed,
    args.rnn_hs, args.pred_hs, args.n_pred_layer, args.learning_rate,
    args.l2, args.n_epoch, args.batch_size, args.patience, True, sw,
    count_path, model_path, args.bound_in, args.pack_all, args.dynamicPT,
    args.max_entry_ratio, args.batch_norm,
)
if BOLT_PROFILE_ENABLED:
    bolt_profile_startup_timings["model_init_and_set_params"] = (
        time.perf_counter() - _t0
    )

model.lambda_order = lambda_order
model.order_tolerance = order_tolerance
model.chain_index = chain_index
model.force_chain_training = args.force_chain_training
model.bolt_profile_startup_timings = bolt_profile_startup_timings
model.bolt_profile_context = {
    "data_name": data_name,
    "workload": workload,
    "chain_index_path": chain_index_path,
}

if not MODE_EST:
    torch.save(model.state_dict(), model_path)

_t0 = time.perf_counter() if BOLT_PROFILE_ENABLED else None
build_time_count = model.build_count(data_name)
if BOLT_PROFILE_ENABLED:
    bolt_profile_startup_timings["main_pre_build_count"] = time.perf_counter() - _t0
efficiency_timer.mark(
    "model_init_chain_and_count_table",
    active_nodes=chain_sampling_stats.get("active_unique_patterns", len(train_data)),
    active_edges=chain_sampling_stats.get("active_pairs_after", 0),
)
model_size = model.model_size()
model_size_hr = ut.human_readable_size(model_size)
print(f"{model_size = }, {model_size_hr = }")
print(f"lambda_order = {lambda_order}")
print(f"order_tolerance = {order_tolerance}")

# ---- 训练或加载 ----
if MODE_EST:
    model.load_state_dict(torch.load(model_path))
    model.n_db = n_db
    model.cuda()
    with open(stat_path) as f:
        build_time = yaml.load(f, Loader=yaml.FullLoader)['time']
else:
    build_time = model.build(data_name, db, train_data, valid_data, test_data)
efficiency_timer.mark("train_or_model_load_with_bounds_and_monotonicity")

print(f"{build_time = }")

# ---- 测试 ----
test_queries, test_cards = list(zip(*test_data))

model.eval()
model.cpu()

start_time = time.time()
with torch.no_grad():
    test_estimations = model.estimate(test_queries, to_cuda=False, is_tqdm=True)
end_time = time.time()
est_time = end_time - start_time
efficiency_timer.mark("test_estimation", queries=len(test_queries))

print(f"{est_time = }")
print(f"{test_estimations[:5] = }")
print(f"{test_cards[:5] = }")

model_size = model.model_size()
print(f"{model_size = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction="none")
mono_score = ""
mono_satisfied = ""
mono_total = ""
t_mono_score = ""
t_mono_satisfied = ""
t_mono_total = ""
chain_pair_score = ""
chain_pair_satisfied = ""
chain_pair_total = ""
chain_all_score = ""
chain_all_satisfied = ""
chain_all_total = ""
chain_pair_score_t = ""
chain_pair_satisfied_t = ""
chain_pair_total_t = ""
chain_all_score_t = ""
chain_all_satisfied_t = ""
chain_all_total_t = ""

# ---- Q-error summary ----
print(f"\n{'=' * 60}")
print(f"Q-ERROR SUMMARY")
print(f"{'=' * 60}")
print(f"  avg:  {np.mean(q_errs):.4f}")
print(f"  q50:  {np.percentile(q_errs, 50):.4f}")
print(f"  q75:  {np.percentile(q_errs, 75):.4f}")
print(f"  q90:  {np.percentile(q_errs, 90):.4f}")
print(f"  q95:  {np.percentile(q_errs, 95):.4f}")
print(f"  q99:  {np.percentile(q_errs, 99):.4f}")
print(f"  q100: {np.max(q_errs):.4f}")
print(f"{'=' * 60}")

# ---- Worst 20 ----
print(f"\n{'=' * 80}")
print(f"TOP 20 WORST PREDICTIONS (by q-error)")
print(f"{'=' * 80}")
print(f"{'Rank':>4s}  {'q-error':>10s}  {'True':>10s}  {'Estimated':>12s}  "
      f"{'UB':>8s}  {'LB':>8s}  Query")
print(f"{'-'*4}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*40}")

worst_indices = np.argsort(q_errs)[::-1][:20]
for rank, idx in enumerate(worst_indices, 1):
    query = test_queries[idx]
    true_card = test_cards[idx]
    est_card = test_estimations[idx]
    qe = q_errs[idx]
    try:
        ub = model.get_bound(query)
        lb = model.get_lower_bound_for_query(query)
    except Exception:
        ub, lb = -1, -1
    print(f"{rank:4d}  {qe:10.2f}  {true_card:10d}  {est_card:12.2f}  "
          f"{ub:8d}  {lb:8d}  {query}")

print(f"{'=' * 80}\n")

# ---- 保存 ----
ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)

count_size = os.path.getsize(model.count_path)
neural_size = os.path.getsize(model.model_path)
ut.save_estimator_stat(stat_path, q_errs, build_time,
                       est_time, model_size=model_size,
                       count_size=count_size, neural_size=neural_size)

with open(stat_path) as f:
    stat_dict = yaml.safe_load(f)
stat_dict.update({
    "force_chain_training": bool(args.force_chain_training),
    "training_path": model.training_path,
    "chain_sample_ratio": float(args.chain_sample_ratio),
    "chain_sampling_policy": args.chain_sampling_policy,
    "checkpoint_selection": "validation_q_error",
})
with open(stat_path, "w") as f:
    yaml.safe_dump(stat_dict, f)

# ---- MonoM Score ----
if chain_index is not None and not SKIP_LEGACY_ORDER_EVAL:
    print(f"\nComputing MonoM score...")
    model.eval()
    mono_score, satisfied, total = compute_mono_score(
        model, list(test_queries), list(test_estimations),
        degrade_steps=3, seed=42,
    )
    print(f"  MonoM score: {mono_score:.4f} ({satisfied}/{total})")

    print(f"\nComputing Chain-MonoM score on test chains...")
    (
        chain_pair_score, chain_pair_satisfied, chain_pair_total,
        chain_all_score, chain_all_satisfied, chain_all_total,
    ) = compute_chain_mono_score(
        model,
        test_queries=list(test_queries),
        max_chain_length=20,
        compress_to=10,
        seed=42,
    )
    print(f"  Chain pair MonoM: {chain_pair_score:.4f} ({chain_pair_satisfied}/{chain_pair_total})")
    print(f"  Chain all MonoM:  {chain_all_score:.4f} ({chain_all_satisfied}/{chain_all_total})")

    print(f"\nComputing tolerance-aware MonoM score (tolerance={order_tolerance})...")
    t_mono_score, t_satisfied, t_total = compute_mono_score(
        model, list(test_queries), list(test_estimations),
        degrade_steps=3, seed=42, tolerance=order_tolerance,
    )
    (
        chain_pair_score_t, chain_pair_satisfied_t, chain_pair_total_t,
        chain_all_score_t, chain_all_satisfied_t, chain_all_total_t,
    ) = compute_chain_mono_score(
        model,
        test_queries=list(test_queries),
        max_chain_length=20,
        compress_to=10,
        seed=42,
        tolerance=order_tolerance,
    )
    t_mono_satisfied, t_mono_total = t_satisfied, t_total
    print(f"  T-MonoM score: {t_mono_score:.4f} ({t_satisfied}/{t_total})")
    print(f"  Chain pair MonoM-T: {chain_pair_score_t:.4f} ({chain_pair_satisfied_t}/{chain_pair_total_t})")
    print(f"  Chain all MonoM-T:  {chain_all_score_t:.4f} ({chain_all_satisfied_t}/{chain_all_total_t})")

    with open(stat_path) as f:
        stat_dict = yaml.safe_load(f)
    stat_dict["mono_score"] = float(mono_score)
    stat_dict["mono_satisfied"] = satisfied
    stat_dict["mono_total"] = total
    stat_dict["chain_pair_mono_score"] = float(chain_pair_score)
    stat_dict["chain_pair_mono_satisfied"] = chain_pair_satisfied
    stat_dict["chain_pair_mono_total"] = chain_pair_total
    stat_dict["chain_all_mono_score"] = float(chain_all_score)
    stat_dict["chain_all_mono_satisfied"] = chain_all_satisfied
    stat_dict["chain_all_mono_total"] = chain_all_total
    stat_dict["t_mono_score"] = float(t_mono_score)
    stat_dict["t_mono_satisfied"] = t_mono_satisfied
    stat_dict["t_mono_total"] = t_mono_total
    stat_dict["chain_pair_mono_score_t"] = float(chain_pair_score_t)
    stat_dict["chain_pair_mono_satisfied_t"] = chain_pair_satisfied_t
    stat_dict["chain_pair_mono_total_t"] = chain_pair_total_t
    stat_dict["chain_all_mono_score_t"] = float(chain_all_score_t)
    stat_dict["chain_all_mono_satisfied_t"] = chain_all_satisfied_t
    stat_dict["chain_all_mono_total_t"] = chain_all_total_t
    with open(stat_path, "w") as f:
        yaml.safe_dump(stat_dict, f)
    print(f"  Saved to {stat_path}")
elif SKIP_LEGACY_ORDER_EVAL:
    print("Skipping legacy generated-chain MonoM; use unified held-out-chain postprocessing.")

ut.append_experiment_summary(
    data_name=data_name,
    workload=workload,
    method=(
        "BOLT_L0" if args.force_chain_training
        else "UB_LB" if lambda_order == 0
        else "BOLT"
    ),
    tag=args.tag,
    seed=seed,
    train_data=train_data,
    valid_data=valid_data,
    test_data=test_data,
    q_errs=q_errs,
    build_time=build_time,
    est_time=est_time,
    model_size=model_size,
    count_size=count_size,
    neural_size=neural_size,
    stat_path=stat_path,
    est_path=est_path,
    lambda_order=lambda_order,
    lb_min_substr_len=args.lb_min_substr_len,
    lb_max_substr_len=args.lb_max_substr_len,
    lb_top_k_strings=args.lb_top_k_strings,
    lb_top_k_substrings=args.lb_top_k_substrings,
    lb_min_freq_ratio=args.lb_min_freq_ratio,
    lower_bound_summary=model.lower_bound_summary,
    mono_score=mono_score,
    mono_satisfied=mono_satisfied,
    mono_total=mono_total,
    chain_pair_mono_score=chain_pair_score,
    chain_pair_mono_satisfied=chain_pair_satisfied,
    chain_pair_mono_total=chain_pair_total,
    chain_all_mono_score=chain_all_score,
    chain_all_mono_satisfied=chain_all_satisfied,
    chain_all_mono_total=chain_all_total,
    t_mono_score=t_mono_score,
    t_mono_satisfied=t_mono_satisfied,
    t_mono_total=t_mono_total,
    chain_pair_mono_score_t=chain_pair_score_t,
    chain_pair_mono_satisfied_t=chain_pair_satisfied_t,
    chain_pair_mono_total_t=chain_pair_total_t,
    chain_all_mono_score_t=chain_all_score_t,
    chain_all_mono_satisfied_t=chain_all_satisfied_t,
    chain_all_mono_total_t=chain_all_total_t,
)
efficiency_timer.mark("result_persistence_complete")
