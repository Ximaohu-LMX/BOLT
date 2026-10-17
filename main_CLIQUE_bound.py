from src.util import read_strings
import src.util as ut
import os
import yaml
import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from src.CLIQUE_bound import CLIQUEestimator_bound
from src.phase_timing import timer as efficiency_timer

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
parser.add_argument(
    "--force_zero_lb", action="store_true",
    help="Force every effective lower bound to zero for the BOUND-LB0 control",
)
args = parser.parse_args()


max_entry_ratio_dict = {
    "WIKI": 4.35,
    "IMDB": 1.675,
    "DBLP": 3.12,
    "GENE": 204.8,
    "AUTHOR": 4.8,
    "DBLP-AN": 3.15,
    "IMDb-AN": 2.6,
    "IMDb-MT": 1.6,
    "TPCH-PN": 15.0,
    "link_type.link": 8.0,
    "movie_companies.note": 0.99,
    "name.name": 0.359,
    "title.title": 0.223,
}
rnn_hs_dict = {
    "DBLP-AN": 160,
    "IMDb-AN": 160,
    "IMDb-MT": 160,
    "TPCH-PN": 160,
    "link_type.link": 181,
    "movie_companies.note": 154,
    "name.name": 152,
    "title.title": 151,
}
if args.data_name in max_entry_ratio_dict:
    args.max_entry_ratio = max_entry_ratio_dict[args.data_name]
if args.data_name in rnn_hs_dict:
    args.rnn_hs = rnn_hs_dict[args.data_name]

args.model_name = f"CLIQUE"
args = ut.add_just_params(args)
args.model_name += "-BOUND-LB0" if args.force_zero_lb else "-BOUND"

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
bound_in = args.bound_in  # bool, whether the model takes bounds as input or not
dynamicPT = args.dynamicPT
max_entry_ratio = args.max_entry_ratio
batch_norm = args.batch_norm
pack_all = args.pack_all
crh = args.crh
tag = args.tag
print(args)

last_flip = True

(
    db_path,
    model_config_path,
    model_path,
    est_path,
    est_time_path,
    stat_path,
) = ut.get_common_pathes_for_estimator(data_name, tag, workload, seed)

sw = ut.get_summary_writer(data_name, tag, workload, seed)
count_path = ut.get_count_path_for_HybridEstimator(
    data_name, PT, N, dynamicPT, max_entry_ratio
)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db = read_strings(db_path)
n_db = len(db)
char_dict, n_special_char = ut.gen_char_dict(db)


char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, p_train)
efficiency_timer.mark("startup_and_data_load")

qc_dict = None
if packed:
    qc_dict = ut.get_qc_dict(data_name, workload)


def packed_card(train_data, qc_dict, last_flip):
    output = []
    for query, card in tqdm(train_data):
        query_cano = ut.canonicalize_like_query(query, last_flip)
        cards = []
        for query_len in range(1, len(query) + 1):
            sub_query = query_cano[:query_len]
            sub_query = ut.canonicalize_like_query(sub_query)
            # sub_query = query[:query_len]
            card = qc_dict[sub_query]
            cards.append(card)
        output.append([query, cards])
    return output


if packed:
    train_data_old = train_data
    train_data = packed_card(train_data, qc_dict, last_flip)

model = CLIQUEestimator_bound()
model.lb_min_substr_len = args.lb_min_substr_len
model.lb_max_substr_len = args.lb_max_substr_len
model.lb_top_k_strings = args.lb_top_k_strings
model.lb_top_k_substrings = args.lb_top_k_substrings
model.lb_min_freq_ratio = args.lb_min_freq_ratio
model.force_zero_lower_bound = args.force_zero_lb
model.use_lower_bound = not args.force_zero_lb
model.set_params(
    N,
    PT,
    seed,
    char_dict,
    ch_es,
    1,
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
)


if not MODE_EST:
    torch.save(model.state_dict(), model_path)
build_time_count = model.build_count(data_name)
efficiency_timer.mark(
    "model_init_and_count_table", active_nodes=len(train_data), active_edges=0
)
model_size = model.model_size()
model_size_hr = ut.human_readable_size(model_size)
print(f"{model_size = }, {model_size_hr = }")


if MODE_EST:
    model.load_state_dict(torch.load(model_path))
    model.n_db = len(db)
    model.cuda()
    with open(stat_path) as f:
        build_time = yaml.load(f, Loader=yaml.FullLoader)['time']
    print(f"{build_time = }")
else:
    build_time = model.build(data_name, db, train_data, valid_data, test_data)
efficiency_timer.mark("train_or_model_load_with_lower_bound")

print(f"{build_time = }")

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

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimation_times(
    test_queries, test_cards, test_estimations, test_estimation_times, est_time_path
)

# save stat (avg, q001, ..., q100, time) # train_time, test_time
count_size = os.path.getsize(model.count_path)
neural_size = os.path.getsize(model.model_path)

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

ut.save_estimator_stat(stat_path, q_errs, build_time,
                       est_time, model_size=model_size,
                       count_size=count_size, neural_size=neural_size)

# ---- MonoM Score ----
mono_score = ""
satisfied = ""
total = ""
if not SKIP_LEGACY_MONO_EVAL:
    print(f"\nComputing MonoM score...")
    model.eval()
    from src.util import compute_mono_score
    mono_score, satisfied, total = compute_mono_score(
        model, list(test_queries), list(test_estimations),
        degrade_steps=3, seed=42,
    )
    print(f"  MonoM score: {mono_score:.4f} ({satisfied}/{total})")
else:
    print("Skipping legacy point-generated MonoM; use unified held-out-chain postprocessing.")

ut.append_experiment_summary(
    data_name=data_name,
    workload=workload,
    method="BOUND_LB0" if args.force_zero_lb else "BOUND",
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
    lb_min_substr_len=args.lb_min_substr_len,
    lb_max_substr_len=args.lb_max_substr_len,
    lb_top_k_strings=args.lb_top_k_strings,
    lb_top_k_substrings=args.lb_top_k_substrings,
    lb_min_freq_ratio=args.lb_min_freq_ratio,
    lower_bound_summary=model.lower_bound_summary,
    mono_score=mono_score,
    mono_satisfied=satisfied,
    mono_total=total,
)
efficiency_timer.mark("result_persistence_complete")
