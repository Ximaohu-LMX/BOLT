from src.util import read_strings
import src.util as ut
import os
import time
import json
import hashlib
import yaml
import torch
import numpy as np

from src.CLIQUE_bound import CLIQUEestimator_bound
from src.CLIQUE_bound_mono import CLIQUEestimator_bound_mono
from src.util import compute_chain_mono_score, compute_mono_score, load_chain_index


MODE_EST = str(os.getenv("MODE")).lower() == "est"
SKIP_LEGACY_MONO_EVAL = str(os.getenv("SKIP_LEGACY_MONO_EVAL", "")).lower() in {
    "1", "true", "yes", "on",
}

input_config_path = "configs/CLIQUE/default.yml"
default_cfgs = yaml.safe_load(open(input_config_path))
abbr_dict = ut.get_common_abbr_dict()
choices_dict = ut.get_common_choices_dict()

parser = ut.get_parser(default_cfgs, abbr_dict, choices_dict)
parser.add_argument(
    "--zero-variant",
    "--zero_variant",
    dest="zero_variant",
    choices=["LB", "MID", "MID_MONO", "ANCHORED_MID_MONO"],
    required=True,
    help="True zero-label variant to run",
)
parser.add_argument(
    "--selection-policy", "--selection_policy",
    dest="selection_policy", choices=["train_mono"], default="train_mono",
    help="Checkpoint policy for trained scale-boundary controls",
)
parser.add_argument("--lambda_mono", type=float, default=5.0,
                    help="Weight for monotonicity loss")
parser.add_argument("--mono_tolerance", type=float, default=0.1,
                    help="Relative tolerance for monotonicity loss dead-zone")
parser.add_argument("--chain_index", type=str, default=None,
                    help="Path to chain_index.json")
parser.add_argument("--lb_min_substr_len", type=int, default=5,
                    help="Minimum frequent substring length for lower bounds")
parser.add_argument("--lb_max_substr_len", type=int, default=15,
                    help="Maximum frequent substring length for lower bounds")
parser.add_argument("--lb_top_k_strings", type=int, default=500,
                    help="Number of frequent complete strings for lower bounds")
parser.add_argument("--lb_top_k_substrings", type=int, default=200,
                    help="Number of frequent substrings per category for lower bounds")
parser.add_argument("--lb_min_freq_ratio", type=float, default=0.0001,
                    help="Minimum frequency ratio for frequent substrings")
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

variant = args.zero_variant
method = (
    "ANCHORED_MID_MONO"
    if variant == "ANCHORED_MID_MONO"
    else f"ZERO_{variant}"
)
args.model_name = "CLIQUE"
args = ut.add_just_params(args)
args.model_name += f"-{method.replace('_', '-')}"
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

data_name = args.data_name
workload = args.workload
seed = args.seed

(
    db_path,
    model_config_path,
    model_path,
    est_path,
    est_time_path,
    stat_path,
) = ut.get_common_pathes_for_estimator(data_name, args.tag, workload, seed)

sw = ut.get_summary_writer(data_name, args.tag, workload, seed)
count_path = ut.get_count_path_for_HybridEstimator(
    data_name, args.PT, args.N, args.dynamicPT, args.max_entry_ratio
)

with open(model_config_path, "w") as f:
    yaml.safe_dump(vars(args), f)

db = read_strings(db_path)
n_db = len(db)
char_dict, n_special_char, train_data, valid_data, test_data = ut.get_training_data(
    data_name, workload, args.p_train
)
labeled_train_rows = sum(1 for _, card in train_data if card >= 0)
if variant == "MID_MONO" and labeled_train_rows != 0:
    raise ValueError(
        f"ZERO MID_MONO requires zero exact training labels; found {labeled_train_rows}"
    )
if variant == "ANCHORED_MID_MONO" and labeled_train_rows == 0:
    raise ValueError("ANCHORED_MID_MONO requires at least one exact training label")


def configure_bound_model(model):
    model.lb_min_substr_len = args.lb_min_substr_len
    model.lb_max_substr_len = args.lb_max_substr_len
    model.lb_top_k_strings = args.lb_top_k_strings
    model.lb_top_k_substrings = args.lb_top_k_substrings
    model.lb_min_freq_ratio = args.lb_min_freq_ratio
    model.set_params(
        args.N,
        args.PT,
        seed,
        char_dict,
        args.ch_es,
        1,
        args.packed,
        args.rnn_hs,
        args.pred_hs,
        args.n_pred_layer,
        args.learning_rate,
        args.l2,
        args.n_epoch,
        args.batch_size,
        args.patience,
        True,
        sw,
        count_path,
        model_path,
        args.bound_in,
        args.pack_all,
        args.dynamicPT,
        args.max_entry_ratio,
        args.batch_norm,
    )
    model.n_db = n_db
    return model


def build_count_and_lower_bound(model):
    build_time_count = model.build_count(data_name)
    start_time = time.time()
    if model.use_lower_bound:
        model.build_lower_bound_table(
            data_name,
            db,
            top_k_strings=args.lb_top_k_strings,
            top_k_substrings=args.lb_top_k_substrings,
            min_substr_len=args.lb_min_substr_len,
            max_substr_len=args.lb_max_substr_len,
            min_freq_ratio=args.lb_min_freq_ratio,
            full_scan=getattr(model, "full_scan", True),
        )
    build_time = build_time_count + (time.time() - start_time)
    return build_time


def size_or_zero(path):
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def effective_lb_ub(model, query):
    ub = float(model.get_bound(query))
    if hasattr(model, "get_effective_lower_bound_for_query"):
        lb = float(model.get_effective_lower_bound_for_query(query, ub))
    else:
        lb = float(model.get_lower_bound_for_query(query))
        lb = min(lb, ub)
        if lb <= 0:
            lb = min(1.0, ub)
    return lb, ub


def geometric_midpoint(lb, ub):
    if ub <= lb:
        return lb
    lb_eff = max(1.0, lb)
    return np.sqrt((lb_eff + 1.0) * (ub + 1.0)) - 1.0


def estimate_static(model, test_queries, static_variant):
    estimates = []
    for query in test_queries:
        lb, ub = effective_lb_ub(model, query)
        if static_variant == "LB":
            pred = lb
        elif static_variant == "MID":
            pred = geometric_midpoint(lb, ub)
        else:
            raise ValueError(static_variant)
        estimates.append(float(max(pred, 0.0)))
    return estimates


def find_last_linear(module):
    last_linear = None
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            last_linear = child
    if last_linear is None:
        raise RuntimeError("Could not find final Linear layer for midpoint init")
    return last_linear


def init_to_geometric_midpoint(model, data):
    bounded = model.append_bound(data, is_tqdm=False, split_name=None)
    ratios = []
    eps = 1e-6
    for _, _, ub, lb in bounded:
        ub = float(ub)
        lb = float(min(lb, ub))
        if ub <= lb:
            continue
        mid = geometric_midpoint(lb, ub)
        ratio = (mid - lb) / (ub - lb)
        ratios.append(float(np.clip(ratio, eps, 1.0 - eps)))

    if ratios:
        target_ratio = float(np.median(ratios))
    else:
        target_ratio = 0.5
    target_logit = float(np.log(target_ratio / (1.0 - target_ratio)))

    final_linear = find_last_linear(model.pred_layer)
    modules = list(model.pred_layer.children())
    pre_activation_bias = target_logit
    if modules and isinstance(modules[-1], torch.nn.LeakyReLU) and target_logit < 0:
        pre_activation_bias = target_logit / modules[-1].negative_slope

    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.fill_(pre_activation_bias)

    state_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        state_digest.update(name.encode() + b'\0')
        state_digest.update(str(tensor.dtype).encode() + b'\0')
        state_digest.update(str(tuple(tensor.shape)).encode() + b'\0')
        state_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    model.midpoint_initialization = {
        'median_ratio': target_ratio, 'target_logit': target_logit,
        'linear_bias': pre_activation_bias, 'nondegenerate_bounds': len(ratios),
        'ratios_sha256': hashlib.sha256(np.asarray(ratios, dtype='<f8').tobytes()).hexdigest(),
        'initial_state_sha256': state_digest.hexdigest(),
        'model_seed': seed, 'rule': 'median shifted geometric midpoint interval position',
    }
    with open(os.path.join(os.path.dirname(model_config_path), 'initialization.json'), 'w') as f:
        json.dump(model.midpoint_initialization, f, indent=2)
    print(
        "ZERO geometric-midpoint init: "
        f"median_ratio={target_ratio:.6f}, "
        f"target_logit={target_logit:.6f}, "
        f"linear_bias={pre_activation_bias:.6f}, "
        f"n_bounds={len(ratios)}"
    )


def print_q_summary(q_errs):
    print(f"\n{'=' * 60}")
    print("Q-ERROR SUMMARY")
    print(f"{'=' * 60}")
    print(f"  avg:  {np.mean(q_errs):.4f}")
    print(f"  q50:  {np.percentile(q_errs, 50):.4f}")
    print(f"  q75:  {np.percentile(q_errs, 75):.4f}")
    print(f"  q90:  {np.percentile(q_errs, 90):.4f}")
    print(f"  q95:  {np.percentile(q_errs, 95):.4f}")
    print(f"  q99:  {np.percentile(q_errs, 99):.4f}")
    print(f"  q100: {np.max(q_errs):.4f}")
    print(f"{'=' * 60}\n")


test_queries, test_cards = list(zip(*test_data))
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

if variant in {"LB", "MID"}:
    model = configure_bound_model(CLIQUEestimator_bound())
    build_time = build_count_and_lower_bound(model)
    model.record_lower_bound_coverage("train", train_data)
    model.record_lower_bound_coverage("valid", valid_data)
    model.record_lower_bound_coverage("test", test_data)

    start_time = time.time()
    test_estimations = estimate_static(model, test_queries, variant)
    est_time = time.time() - start_time
    model_size = 0
    count_size = size_or_zero(model.count_path)
    neural_size = 0

elif variant in {"MID_MONO", "ANCHORED_MID_MONO"}:
    chain_index_path = args.chain_index
    if chain_index_path is None:
        chain_index_path = f"data/{data_name}/training/{workload}/chain_index.json"
    if not os.path.exists(chain_index_path):
        raise FileNotFoundError(f"Missing chain_index.json: {chain_index_path}")

    chain_index = load_chain_index(chain_index_path)
    print(f"Loaded chain index: {len(chain_index)} chains from {chain_index_path}")

    model = configure_bound_model(CLIQUEestimator_bound_mono())
    model.lambda_mono = args.lambda_mono
    model.mono_tolerance = args.mono_tolerance
    model.chain_index = chain_index
    # Both trained B4 rows use exactly the same target-free checkpoint rule.
    # Endpoint labels affect gradients in the anchored row, not model selection.
    model.label_free_selection = args.selection_policy == "train_mono"

    build_time = build_count_and_lower_bound(model)
    init_to_geometric_midpoint(model, train_data)

    if MODE_EST:
        model.load_state_dict(torch.load(model_path))
        model.n_db = n_db
        model.cuda()
        with open(stat_path) as f:
            build_time = yaml.load(f, Loader=yaml.FullLoader)["time"]
    else:
        build_time += model.build_model(train_data, valid_data, test_data)

    model.eval()
    model.cpu()
    start_time = time.time()
    with torch.no_grad():
        test_estimations = model.estimate(test_queries, to_cuda=False, is_tqdm=True)
    est_time = time.time() - start_time
    model_size = model.model_size()
    count_size = size_or_zero(model.count_path)
    neural_size = size_or_zero(model.model_path)

    if SKIP_LEGACY_MONO_EVAL:
        print(
            "Skipping legacy generated-chain MonoM; use unified "
            "held-out-chain postprocessing."
        )
    else:
        print("\nComputing MonoM score...")
        mono_score, mono_satisfied, mono_total = compute_mono_score(
            model, list(test_queries), list(test_estimations), degrade_steps=3, seed=42
        )
        print(f"  MonoM score: {mono_score:.4f} ({mono_satisfied}/{mono_total})")
        (
            chain_pair_score,
            chain_pair_satisfied,
            chain_pair_total,
            chain_all_score,
            chain_all_satisfied,
            chain_all_total,
        ) = compute_chain_mono_score(
            model,
            test_queries=list(test_queries),
            max_chain_length=20,
            compress_to=10,
            seed=42,
        )
        print(f"  Chain pair MonoM: {chain_pair_score:.4f} ({chain_pair_satisfied}/{chain_pair_total})")
        print(f"  Chain all MonoM:  {chain_all_score:.4f} ({chain_all_satisfied}/{chain_all_total})")

        t_mono_score, t_mono_satisfied, t_mono_total = compute_mono_score(
            model,
            list(test_queries),
            list(test_estimations),
            degrade_steps=3,
            seed=42,
            tolerance=args.mono_tolerance,
        )
        (
            chain_pair_score_t,
            chain_pair_satisfied_t,
            chain_pair_total_t,
            chain_all_score_t,
            chain_all_satisfied_t,
            chain_all_total_t,
        ) = compute_chain_mono_score(
            model,
            test_queries=list(test_queries),
            max_chain_length=20,
            compress_to=10,
            seed=42,
            tolerance=args.mono_tolerance,
        )
        print(f"  T-MonoM score: {t_mono_score:.4f} ({t_mono_satisfied}/{t_mono_total})")
        print(f"  Chain pair MonoM-T: {chain_pair_score_t:.4f} ({chain_pair_satisfied_t}/{chain_pair_total_t})")
        print(f"  Chain all MonoM-T:  {chain_all_score_t:.4f} ({chain_all_satisfied_t}/{chain_all_total_t})")
else:
    raise ValueError(variant)

print(f"{est_time = }")
print(f"{test_estimations[:5] = }")
print(f"{test_cards[:5] = }")

q_errs = ut.mean_Q_error(test_cards, test_estimations, reduction="none")
print_q_summary(q_errs)

ut.save_estimated_cards(test_queries, test_cards, test_estimations, est_path)
ut.save_estimator_stat(
    stat_path,
    q_errs,
    build_time,
    est_time,
    model_size=model_size,
    count_size=count_size,
    neural_size=neural_size,
)

with open(stat_path) as f:
    stat_dict = yaml.safe_load(f)
stat_dict.update({
    "b4_variant": method,
    "midpoint_initialization": getattr(model, 'midpoint_initialization', None),
    "best_epoch": getattr(model, 'selection_best_epoch', None),
    "best_train_mono": getattr(model, 'selection_best_score', None),
    "epochs_completed": getattr(model, 'selection_epochs_completed', None),
    "timing_usable": False,
    "actual_training_pattern_order_sha256": hashlib.sha256(
        ''.join(q+'\n' for q,c in train_data).encode()).hexdigest(),
    "actual_training_labels_sha256": hashlib.sha256(
        ''.join(f'{q},{int(c)}\n' for q,c in train_data if c>=0).encode()).hexdigest(),
    "train_labeled": int(labeled_train_rows),
    "train_unlabeled": int(len(train_data) - labeled_train_rows),
    "initialization_policy": (
        "geometric_log_midpoint" if "MID_MONO" in variant else "static"
    ),
    "checkpoint_selection": (
        args.selection_policy if "MID_MONO" in variant else "not_applicable"
    ),
    "exact_validation_targets_used_for_selection": False,
})
with open(stat_path, "w") as f:
    yaml.safe_dump(stat_dict, f)

if variant in {"MID_MONO", "ANCHORED_MID_MONO"} and not SKIP_LEGACY_MONO_EVAL:
    with open(stat_path) as f:
        stat_dict = yaml.safe_load(f)
    stat_dict["mono_score"] = float(mono_score)
    stat_dict["mono_satisfied"] = mono_satisfied
    stat_dict["mono_total"] = mono_total
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

ut.append_experiment_summary(
    data_name=data_name,
    workload=workload,
    method=method,
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
    lambda_mono=(
        args.lambda_mono
        if variant in {"MID_MONO", "ANCHORED_MID_MONO"}
        else ""
    ),
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
