from collections import defaultdict


def getNF(enabledness_model, traces, always_enabled = None):
    if always_enabled is None:
        always_enabled = set()

    never_followed_candidates = defaultdict(set)
    for context, enabledness in enabledness_model.items():
        p1_disabled_tokens = set()
        for token, prob in enabledness.items():
            if prob <= 0.0 and "EXCEPTION" not in token and token not in always_enabled:
                p1_disabled_tokens.add(token)
        for p1_disabled_token in p1_disabled_tokens:
            print("LTL:NIF", context[0], p1_disabled_token)
            never_followed_candidates[context[0]].add(p1_disabled_token)

    for trace in traces:
        candidate_trace_of = set()
        for event in trace:

            if isinstance(event, str):
                event_str = event
            else:
                event_str = event[0] + (":" + event[1] if event[1] is not None else "")

            if event_str in never_followed_candidates:
                candidate_trace_of.add(event_str)

            if len(candidate_trace_of) == 0:
                continue

            # in candidate trace, which means that we have seen at least one candidate pair
            to_remove = defaultdict(set)
            for candidate in candidate_trace_of:
                for closing_candidate in never_followed_candidates[candidate]:
                    if closing_candidate == event_str:
                        to_remove[candidate].add(closing_candidate)
            for key, items in to_remove.items():
                for item in items:
                    never_followed_candidates[key].remove(item)
    for key, values in never_followed_candidates.items():
        for value in values:
            print("LTL:NF", key, value)
    return never_followed_candidates


