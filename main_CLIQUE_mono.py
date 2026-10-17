"""
main_CLIQUE_mono.py  —  带单调性约束的 CLIQUE 训练入口

基于 main_CLIQUE.py，增加：
  1. 加载 chain_index.json 获取链结构
  2. 将链结构传入模型训练，启用单调性 loss
  3. 训练完成后计算 MonoM score

用法:
  python main_CLIQUE_mono.py -d AUTHOR -w SEMO --seed 0 --lambda_mono 1.0

与原 main_CLIQUE.py 的区别:
  - 新增参数: --lambda_mono, --chain_index
  - 训练循环使用 chain-aware batching + 双 loss
  - 测试阶段额外输出 MonoM score
"""

from src.util import read_strings
import src.util as ut
import os
import yaml
import time
import torch
import pickle
import json
import numpy as np
from tqdm import tqdm

# ---- 导入修改后的 CLIQUE 模块 ----
from src.CLIQUE_mono import CLIQUEestimator_mono
from src.phase_timing import timer as efficiency_timer
from src.util import compute_chain_mono_score, compute_mono_score, load_chain_index

# ---- 与原 main_CLIQUE.py 相同的配置加载 ----
MODE_EST = str(os.getenv("MODE")).lower() == "est"
MODE_SIZE = str(os.getenv("MODE")).lower() == "size"
SKIP_LEGACY_MONO_EVAL = str(os.getenv("SKIP_LEGACY_MONO_EVAL", "")).lower() in {
    "1", "true", "yes", "y", "on",
}

input_config_path = "configs/CLIQUE/default.yml"
default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)

# 新增参数
parser.add_argument("--lambda_mono", type=float, default=1.0,
                    help="Weight for monotonicity loss (default: 1.0)")
parser.add_argument("--mono_tolerance", type=float, default=0.0,
                    help="Relative tolerance for monotonicity loss dead-zone")
parser.add_argument("--chain_index", type=str, default=None,
                    help="Path to chain_index.json (auto-detected if not set)")

args = parser.parse_args()

# ---- 与原 main_CLIQUE.py 相同的参数处理 ----
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

args.model_name = f"CLIQUE"
args = ut.add_just_params(args)

if args.packed:
    args.model_name += "-PACK"
if not args.crh:
    args.model_name += "-T"
if args.pack_all:
    args.model_name += "-ALL"
if args.table_only:
    args.model_name = "TABLE"

# 加上 MONO 标记
args.model_name += "-MONO"

args.tag = os.path.join(args.tag, args.model_name)
print(args)

# ---- 参数提取 ----
N = args.N
PT = args.PT
data_name = args.data_name
workload = args.workload
seed = args.seed
n_epoch = args.n_epoch
ch_es = args.ch_es
rnn_hs = args.rnn_hs
pred_hs = args.pred_hs
p_test = args.p_test
p_val = args.p_val
p_train = args.p_train
n_pred_layer = args.n_pred_layer
bs = args.batch_size
lr = args.learning_rate
l2 = args.l2
patience = args.patience
packed = args.packed
bound_in = args.bound_in
dynamicPT = args.dynamicPT
max_entry_ratio = args.max_entry_ratio
batch_norm = args.batch_norm
pack_all = args.pack_all
crh = args.crh
tag = args.tag
lambda_mono = args.lambda_mono
mono_tolerance = args.mono_tolerance

last_flip = True

# ---- 路径 ----
(
    db_path, model_config_path, model_path,
    est_path, est_time_path, stat_path,
) = ut.get_common_pathes_for_estimator(data_name, tag, workload, seed)

sw = ut.get_summary_writer(data_name, tag, workload, seed)
count_path = ut.get_count_path_for_HybridEstimator(
    data_name, PT, N, dynamicPT, max_entry_ratio
)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

# ---- 加载数据 ----
db = read_strings(db_path)
n_db = len(db)
char_dict, n_special_char = ut.gen_char_dict(db)

char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, p_train)
efficiency_timer.mark("startup_and_data_load")

# ---- 加载链索引 ----
chain_index_path = args.chain_index
if chain_index_path is None:
    chain_index_path = f"data/{data_name}/training/{workload}/chain_index.json"

chain_index = None
if os.path.exists(chain_index_path):
    chain_index = load_chain_index(chain_index_path)
    print(f"Loaded chain index: {len(chain_index)} chains from {chain_index_path}")
else:
    print(f"WARNING: chain_index.json not found at {chain_index_path}")
    print(f"  Training without monotonicity constraint (falling back to standard CLIQUE)")
    lambda_mono = 0.0

# ---- packed 模式（当前不与链结合使用） ----
qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)

if packed:
    def packed_card(train_data, qc_dict, last_flip):
        output = []
        for query, card in tqdm(train_data):
            query_cano = ut.canonicalize_like_query(query, last_flip)
            cards = []
            for query_len in range(1, len(query) + 1):
                sub_query = query_cano[:query_len]
                sub_query = ut.canonicalize_like_query(sub_query)
                card = qc_dict[sub_query]
                cards.append(card)
            output.append([query, cards])
        return output
    train_data_old = train_data
    train_data = packed_card(train_data, qc_dict, last_flip)

# ---- 创建模型 ----
model = CLIQUEestimator_mono()
model.set_params(
    N, PT, seed, char_dict, ch_es, 1, packed, rnn_hs, pred_hs,
    n_pred_layer, lr, l2, n_epoch, bs, patience, last_flip, sw,
    count_path, model_path, bound_in, pack_all, dynamicPT,
    max_entry_ratio, batch_norm,
)

# 设置单调性参数
model.lambda_mono = lambda_mono
model.mono_tolerance = mono_tolerance
model.chain_index = chain_index

if not MODE_EST:
    torch.save(model.state_dict(), model_path)
build_time_count = model.build_count(data_name)
efficiency_timer.mark(
    "model_init_chain_and_count_table",
    active_nodes=(
        len({
            pattern
            for chain in chain_index
            for pattern in chain.get("patterns", [])
        }) if chain_index else len(train_data)
    ),
    active_edges=sum(
        max(len(chain.get("patterns", [])) - 1, 0)
        for chain in (chain_index or [])
    ),
)
model_size = model.model_size()
model_size_hr = ut.human_readable_size(model_size)
print(f"{model_size = }, {model_size_hr = }")
print(f"lambda_mono = {lambda_mono}")
print(f"mono_tolerance = {mono_tolerance}")

# ---- 训练或加载 ----
if MODE_EST:
    model.load_state_dict(torch.load(model_path))
    model.n_db = len(db)
    model.cuda()
    with open(stat_path) as f:
        build_time = yaml.load(f, Loader=yaml.FullLoader)['time']
    print(f"{build_time = }")
else:
    build_time = model.build(data_name, db, train_data, valid_data, test_data)
efficiency_timer.mark("train_or_model_load_with_monotonicity")

print(f"{build_time = }")

# ---- 测试 ----
test_queries, test_cards = list(zip(*test_data))

model.eval()
model.cpu()

start_time = time.time()
with torch.no_grad():
    test_estimations, test_estimation_times = model.estimate_latency_analysis(
        test_queries, to_cuda=False,
    )
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

# ---- Worst 20 predictions ----
print(f"\n{'=' * 80}")
print(f"TOP 20 WORST PREDICTIONS (by q-error)")
print(f"{'=' * 80}")
print(f"{'Rank':>4s}  {'q-error':>10s}  {'True':>10s}  {'Estimated':>12s}  {'Bound':>10s}  Query")
print(f"{'-'*4}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*40}")

worst_indices = np.argsort(q_errs)[::-1][:20]
for rank, idx in enumerate(worst_indices, 1):
    query = test_queries[idx]
    true_card = test_cards[idx]
    est_card = test_estimations[idx]
    qe = q_errs[idx]
    # 查上界
    try:
        bound = model.get_bound(query)
    except Exception:
        bound = -1
    print(f"{rank:4d}  {qe:10.2f}  {true_card:10d}  {est_card:12.2f}  {bound:10d}  {query}")

print(f"{'=' * 80}\n")

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
)

count_size = os.path.getsize(model.count_path)
neural_size = os.path.getsize(model.model_path)
ut.save_estimator_stat(stat_path, q_errs, build_time,
                       est_time, model_size=model_size,
                       count_size=count_size, neural_size=neural_size)

# ---- MonoM Score ----
if chain_index is not None and not SKIP_LEGACY_MONO_EVAL:
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

    print(f"\nComputing tolerance-aware MonoM score (tolerance={mono_tolerance})...")
    t_mono_score, t_satisfied, t_total = compute_mono_score(
        model, list(test_queries), list(test_estimations),
        degrade_steps=3, seed=42, tolerance=mono_tolerance,
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
        tolerance=mono_tolerance,
    )
    t_mono_satisfied, t_mono_total = t_satisfied, t_total
    print(f"  T-MonoM score: {t_mono_score:.4f} ({t_satisfied}/{t_total})")
    print(f"  Chain pair MonoM-T: {chain_pair_score_t:.4f} ({chain_pair_satisfied_t}/{chain_pair_total_t})")
    print(f"  Chain all MonoM-T:  {chain_all_score_t:.4f} ({chain_all_satisfied_t}/{chain_all_total_t})")

    # 追加到 stat
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
    print(f"  MonoM scores saved to {stat_path}")
elif SKIP_LEGACY_MONO_EVAL:
    print("Skipping legacy generated-chain MonoM; use unified held-out-chain postprocessing.")

ut.append_experiment_summary(
    data_name=data_name,
    workload=workload,
    method="MONO",
    tag=tag,
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
    lambda_mono=lambda_mono,
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
