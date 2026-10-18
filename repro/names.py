"""Paper method labels and portable identifiers used by every current run."""
LABELS = {
    "BOLT": "BOLT", "CLIQUE": "CLIQUE", "UB_LB": "UB+LB",
    "UB_ORDER": "UB+Order", "LB_ORDER": "LB+Order", "NO_BOUND": "NO-BOUND",
    "BOLT_L0": "BOLT (λ=0)", "ZERO_LB": "Lower bound",
    "ZERO_MID": "Shifted midpoint", "ZERO_MID_ORDER": "Order only",
    "ANCHORED_MID_ORDER": "Order + endpoints", "DREAM": "DREAM", "LPLM": "LPLM", "LBS": "LBS",
}

def method_id(value):
    aliases = {label.casefold(): key for key, label in LABELS.items()}
    aliases.update({key.casefold(): key for key in LABELS})
    aliases.update({"ub": "CLIQUE", "zero-mid-order": "ZERO_MID_ORDER",
                    "anchored-mid-order": "ANCHORED_MID_ORDER", "zero-lb": "ZERO_LB", "zero-mid": "ZERO_MID"})
    try:
        return aliases[value.casefold()]
    except KeyError:
        raise ValueError("Unknown method: " + value + "; use a paper name or a documented identifier") from None

def display_method(value, ablation=False):
    method = method_id(value)
    return "UB" if ablation and method == "CLIQUE" else LABELS[method]
