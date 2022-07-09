import functools
import sys
from collections import defaultdict
import os

from fsa.acceptor import infer_model
from fsa.rule_utils import getNF
from pure.pure_handler import read_pure_functions
from violations.violation_checker import learn_from_traces_for_model
from traces.trace_reader import read_traces
import random


strip_descriptor = True

random.seed(9000)

def main():
    print(sys.argv)
    traces = read_traces(sys.argv[1], strip_descriptor)

    pure_functions = read_pure_functions(sys.argv[2], strip_descriptor) # functions statically analysed to be pure

    if sys.argv[3] != "None":
        additional_traces = read_traces(sys.argv[3], strip_descriptor)
        print('with additional traces')
    else:
        additional_traces = []
        print('without additional traces')

    traces.extend(additional_traces)
    vocab = set()
    can_be_disabled = set()
    ctor = None
    for trace in traces:
        for event in trace:
            if isinstance(event, str):
                event_str = event
            else:
                event_str = event[0]

            if "EXCEPTION" in event_str:
                can_be_disabled.add(event_str)
                continue
            vocab.add(event_str)
            if is_constructor(event_str):
                ctor = event_str
                
    keep_enabled = set()
    if len(can_be_disabled) > 0:
        # if there are exceptional behaviour (e.g. "IO-LEAK", "<exception>" tracked in the traces
        # we don't want to include them in the fsa
        keep_enabled = vocab - set(can_be_disabled) - set(["<START>", "<END>", ctor])

    try:
        # set a lower number here if too slow
        num_traces = int(sys.argv[5])
        print('taking only', num_traces, ' traces')
    except:
        num_traces = 20000

    fsa = infer_fsa(traces, set(pure_functions), num_traces, keep_enabled, additional_traces)

    # with open('latest_output.dot', 'w+') as outfile:
    #     outfile.write(fsa.to_dot())
    #
    # print('writing to latest_output.dot')
    # with open('latest_output.dot', 'w+') as outfile:
    #     outfile.write(fsa.to_dot())
    print('writing to ', sys.argv[4])
    if not os.path.exists(os.path.dirname(sys.argv[4])):
        os.makedirs(os.path.dirname(sys.argv[4]))
    with open(sys.argv[4], 'w+') as outfile:
        outfile.write(fsa.to_evaluation_format())

    fsa.serialize(sys.argv[4] + '.json')

def infer_fsa(traces, pure_functions, num_traces, always_enabled, additional_traces):
    # for implicitly checking some rule types
    enabledness_model = learn_from_traces_for_model(traces, pure_functions)
    reverse_traces = [reversed(trace) for trace in traces]
    reverse_enabledness_model = learn_from_traces_for_model(reverse_traces, pure_functions)


    fsa_traces = random.sample(traces, num_traces if len(traces) > num_traces else len(traces))

    fsa_traces.extend(additional_traces)

    print('inferring model')
    fsa = infer_model(fsa_traces, pure_functions, enabledness_model)
    NFs = getNF(enabledness_model.copy(), traces)

    fsa.assert_global_invariants()
    rules = []
    for NF_eventA, NF_eventBs in NFs.items():
        for NF_eventB in NF_eventBs:
            # print('NF::', NF_eventA, NF_eventB)
            if "<START>" in NF_eventA or "<START>" in NF_eventB:
                continue
            if "<END>" in NF_eventA or "<END>" in NF_eventB:
                continue
            if "<init>" in NF_eventA or "<init>" in NF_eventB:
                continue
            if NF_eventA[0].isupper() or NF_eventB[0].isupper() :
                continue
            rules.append(("never_followed", NF_eventA, NF_eventB))
    print('rules are')
    print(rules)

    with open('before_merging.dot', 'w+') as outfile:
        outfile.write(fsa.to_dot())

    print('merging')
    fsa = merge_states(fsa, pure_functions, k=1,
                       enabledness_model=(enabledness_model, reverse_enabledness_model),
                       merge_rejector=functools.partial(are_temporal_rules_true, rules, pure_functions),
                       always_enabled=always_enabled,
                       nf_rules=[rule for rule in rules if rule[0] == "never_followed"])
    print('merging done')
    # with open('before_cleanup_output.dot', 'w+') as outfile:
    #     outfile.write(fsa.to_dot())

    fsa.assert_global_invariants()
    return fsa




def is_constructor(m):
    return m[0].isalpha() and m[0].isupper() and m[1].isalpha()


def are_temporal_rules_true(rules, pures, fsa, node_prefixes, new_state, node_suffixes):

    for rule in rules:
        rule_type, label1, label2 = rule
        if rule_type == "never_purely_followed":
            pass
        elif rule_type == "never_followed":
            missing_from_all_prefix = True

            for node_prefix in node_prefixes:
                if label1 in node_prefix:
                    missing_from_all_prefix = False
                    break

            if not missing_from_all_prefix and fsa.is_followed(label1, label2, pures, new_state, node_prefix, node_suffixes):
                return True

        elif rule_type == "never_preceded":
            # we don't have to check rule types other than NF
            # these types are implicitly checked when we perform the "enabledness" checks
            # e.g. if a "NP" is true,
            pass
        elif rule_type == "never_purely_preceded":
            pass
        elif rule_type == "always_followed":
            pass
        else:
            raise Exception("wrong rule type. your rule_type is " + rule_type + " and this doesn't exist...")
    return False

def no_change(fsa):
    fsa.done_merge = True

def reverse_inference(backwards_model, impure_observations, pure_observations):
    """
    Given the observations, guess which possible inputs CAN'T have produced it.
    """
    results = set()

    first_tokens_of_impure_observersations = set([events[0] for events in impure_observations if len(events) > 0])

    for context, token_probs in backwards_model.items():
        assert type(context) == tuple
        if context[0] not in first_tokens_of_impure_observersations and context[0] not in pure_observations:
            continue
        for token, probs in token_probs.items():
            if probs < 0:
                results.add(token)
    return results


def get_NF_disabled(p1, prefixes, nf_rules):
    disabled = set()
    for rule in nf_rules:
        rule_type, label1, label2 = rule
        assert rule_type == "never_followed"
        for prefix in prefixes:
            if label1 in prefix:
                disabled.add(label2)


    return disabled

def merge_states(fsa, pures, k=2, enabledness_model=None, merge_rejector=None, always_enabled=None, nf_rules =None):
    """
    similar to k-tails, but with knowledge of methods that are not "enabled",
    and of NF rules that has to be checked
    """
    prefixes = fsa.bfs()
    print('obtained prefixes')

    equal_nodes = set()
    future_seq_of_each_node = {}
    node_index_to_prefix = {}
    node_index_to_pures = {}
    node_index_to_disabled = {}
    node_index_to_disabled_reverse = {}
    compute_k_futures_for_prefixes(fsa, k, future_seq_of_each_node, node_index_to_prefix, node_index_to_pures, prefixes,
                                   pures)
    forward_enabledness_model, backwards_enabled_model = enabledness_model

    nodes = future_seq_of_each_node.keys()
    seen = set()

    unique_disabledness = set()
    counter = 0

    print('nodes' ,len(nodes))
    for p1 in nodes:
        if p1 in seen:
            continue
        seen.add(p1)

        k_future_p1s = future_seq_of_each_node[p1]
        for k_future_p1_seq, k_future_p1_pures in k_future_p1s:
            k_seq_only_p1 = set(k_future_p1_seq)
            k_pure_funcs_p1 = set([pure_func for pure_func in k_future_p1_pures])

            p1_disabled_tokens = set()
            node_index_to_disabled[p1] = p1_disabled_tokens

            if node_index_to_prefix[p1] in forward_enabledness_model:

                p1_enabledness = forward_enabledness_model[node_index_to_prefix[p1]]
                for token, prob in p1_enabledness.items():
                    if prob < 0.0 and "EXCEPTION" not in token and token not in always_enabled:
                        p1_disabled_tokens.add(token)

                p1_reverse_direction_disabled_tokens = reverse_inference(backwards_enabled_model, k_seq_only_p1,
                                                                         k_pure_funcs_p1 | node_index_to_pures[p1])
                node_index_to_disabled_reverse[p1] = p1_reverse_direction_disabled_tokens
            for pures_around in k_pure_funcs_p1 | node_index_to_pures[p1]:
                p1_enabledness = forward_enabledness_model[(pures_around,)]

                for token, prob in p1_enabledness.items():
                    if prob < 0.0 and "EXCEPTION" not in token and token not in always_enabled: #and token.split(':')[0] in pures:
                        p1_disabled_tokens.add(token)

            for seq_future in k_seq_only_p1:
                if len(seq_future) >= 1 and seq_future[0] in p1_disabled_tokens:
                    assert False

            disabled_by_NF = get_NF_disabled(p1, fsa.prefixes[p1], nf_rules)

            p1_disabled_tokens.update(disabled_by_NF)

            for p2 in nodes:
                if p2 in seen:
                    continue

                k_future_p2s = future_seq_of_each_node[p2]
                for k_future_p2_seq, k_future_p2_pures in k_future_p2s:
                    k_seq_only_p2 = set(k_future_p2_seq)
                    k_pure_funcs_p2 = set([pure_func for pure_func in k_future_p2_pures])

                    p2_disabled_tokens = set()
                    if p2 in node_index_to_disabled and p2 in node_index_to_disabled_reverse:
                        p2_disabled_tokens = node_index_to_disabled[p2]
                        p2_reverse_direction_disabled_tokens = node_index_to_disabled_reverse[p2]
                    else:
                        node_index_to_disabled[p2] = p2_disabled_tokens

                        if node_index_to_prefix[p2] in forward_enabledness_model:
                            p2_enabledness = forward_enabledness_model[node_index_to_prefix[p2]]
                            for token, prob in p2_enabledness.items():
                                if prob < 0.0 and "EXCEPTION" not in token and token not in always_enabled:
                                    p2_disabled_tokens.add(token)

                            p2_reverse_direction_disabled_tokens = reverse_inference(backwards_enabled_model, k_seq_only_p2,
                                                                                     k_pure_funcs_p2 | node_index_to_pures[p2])
                            node_index_to_disabled_reverse[p2] = p2_reverse_direction_disabled_tokens
                        for pures_around in k_pure_funcs_p2 | node_index_to_pures[p2]:
                            p2_enabledness = forward_enabledness_model[(pures_around, )]
                            for token, prob in p2_enabledness.items():
                                if prob < 0.0 and "EXCEPTION" not in token and token not in always_enabled: # and token.split(':')[0] in pures:
                                    p2_disabled_tokens.add(token)

                        disabled_by_NF = get_NF_disabled(p2, fsa.prefixes[p2], nf_rules)
                        p2_disabled_tokens.update(disabled_by_NF)

                        for seq_future in k_seq_only_p2:
                            if len(seq_future) >= 1 and seq_future[0] in p2_disabled_tokens:
                                assert False

                    if node_index_to_prefix[p1] not in forward_enabledness_model:
                        continue

                    if node_index_to_prefix[p2] not in forward_enabledness_model:
                        continue


                    remove_start_and_end_tokens(p1_disabled_tokens)
                    remove_start_and_end_tokens(p2_disabled_tokens)
                    remove_start_and_end_tokens(p1_reverse_direction_disabled_tokens)
                    remove_start_and_end_tokens(p2_reverse_direction_disabled_tokens)

                    unique_disabledness.add((tuple(p1_disabled_tokens), tuple(p1_reverse_direction_disabled_tokens)))
                    unique_disabledness.add((tuple(p2_disabled_tokens), tuple(p2_reverse_direction_disabled_tokens)))

                    if p1_disabled_tokens == p2_disabled_tokens and \
                            p1_reverse_direction_disabled_tokens == p2_reverse_direction_disabled_tokens:
                        equal_nodes.add((p1, p2))

    i = 0  # just used for debugging
    merges = {}

    original_nodes_of = defaultdict(set)

    failed_merges = set()
    consecutive_failed_merges = 0
    # print('begin checking nodes that may be merged')
    for n1, n2 in equal_nodes:
        i += 1

        if i % 100000 == 0:
            print(i)

        # as the fsa gets mutated within the loop,
        # we keep track of the mappings of new nodes formed by the nodes that are already merged
        n1_in_current_fsa = get_final_nodes(merges, n1)
        n2_in_current_fsa = get_final_nodes(merges, n2)

        if n1_in_current_fsa == n2_in_current_fsa:
            # already same node. skip
            continue
        if (n1, n2) in failed_merges:
            continue

        reversal, new_state = fsa.merge_states(n1_in_current_fsa, n2_in_current_fsa)
        prefixes = []
        for origin_node in original_nodes_of[n1_in_current_fsa]:
            prefixes.extend(fsa.prefixes[origin_node])
        for origin_node in original_nodes_of[n2_in_current_fsa]:
            prefixes.extend(fsa.prefixes[origin_node])

        suffixes = []
        for origin_node in original_nodes_of[n1_in_current_fsa]:
            suffixes.extend(fsa.suffixes[origin_node])
        for origin_node in original_nodes_of[n2_in_current_fsa]:
            suffixes.extend(fsa.suffixes[origin_node])

        if merge_rejector is not None and merge_rejector(fsa, prefixes, new_state, suffixes) is True:
            # rejected!
            reversal()
            failed_merges.add((n1, n2))
            continue

        # merged nodes shouldn't gain new methods that were previously disabled
        if n1_in_current_fsa in node_index_to_disabled and fsa.is_state_have_disabled_methods(new_state, node_index_to_disabled[n1_in_current_fsa]):
            reversal()
            failed_merges.add((n1, n2))
            consecutive_failed_merges += 1
            continue

        if n2_in_current_fsa in node_index_to_disabled and fsa.is_state_have_disabled_methods(new_state, node_index_to_disabled[n2_in_current_fsa]):
            reversal()
            failed_merges.add((n1, n2))
            consecutive_failed_merges += 1
            continue

        if n1 in node_index_to_disabled_reverse and fsa.is_state_have_disabled_methods_backwards(
                new_state, node_index_to_disabled_reverse[n1_in_current_fsa]):
            reversal()
            failed_merges.add((n1, n2))
            consecutive_failed_merges += 1
            continue
        if n2 in node_index_to_disabled_reverse and fsa.is_state_have_disabled_methods_backwards(
                new_state, node_index_to_disabled_reverse[n2_in_current_fsa]):
            reversal()
            failed_merges.add((n1, n2))
            consecutive_failed_merges += 1
            continue

        node1_to_update = n1
        while node1_to_update in merges:
            next_node1_to_update = merges[node1_to_update]
            merges[node1_to_update] = new_state
            node1_to_update  = next_node1_to_update
        merges[node1_to_update] = new_state
        node2_to_update = n2
        while node2_to_update in merges:
            next_node2_to_update = merges[node2_to_update]
            merges[node2_to_update] = new_state
            node2_to_update = next_node2_to_update
        merges[node2_to_update] = new_state

        original_nodes_of[new_state].update(original_nodes_of[n1_in_current_fsa])
        original_nodes_of[new_state].update(original_nodes_of[n2_in_current_fsa])
        original_nodes_of[new_state].add(n1)
        original_nodes_of[new_state].add(n2)

        node_index_to_disabled[new_state] = node_index_to_disabled[n1_in_current_fsa] | node_index_to_disabled[n2_in_current_fsa]
        node_index_to_disabled_reverse[new_state] = node_index_to_disabled_reverse[n1_in_current_fsa] | node_index_to_disabled_reverse[
            n2_in_current_fsa]

        consecutive_failed_merges = 0

        counter += 1

    fsa.done_merging()

    return fsa


def compute_k_futures_for_prefixes(fsa, k, future_seq_of_each_node, node_index_to_prefix, node_index_to_pures, prefixes,
                                   pures):
    for prefix in prefixes:
        p1_futures = fsa.future_seq_splitting_pure_and_ordering(k, prefix, pures)

        assert isinstance(p1_futures, {}.__class__)
        future_seq_of_each_node.update(p1_futures)

        # compute last n impure events in prefix
        impure_prefix = []
        # the pure events seen in the prefix
        pure_set = set()

        for event in reversed(prefix):
            event_without_retval = event.split(':')[0]
            if event_without_retval in pures:
                pure_set.add(event)
                continue

            impure_prefix.append(event)
            if len(impure_prefix) == 1:
                break
        impure_prefix = tuple(reversed(impure_prefix))

        for node in p1_futures.keys():
            node_index_to_prefix[node] = impure_prefix
            node_index_to_pures[node] = pure_set


def remove_start_and_end_tokens(tokens):
    if "<START>" in tokens:
        # print('remove start and ends')
        tokens.remove("<START>")
        # print(tokens)
    if "<END>" in tokens:
        tokens.remove("<END>")


def get_final_nodes(merges, n1):
    if n1 in merges:  # already merged
        n1_in_current_fsa = n1
        while n1_in_current_fsa in merges:
            if n1_in_current_fsa == merges[n1_in_current_fsa]:
                sys.stderr.write("SAME " + str(n1_in_current_fsa) + "\n")
                assert False
            n1_in_current_fsa = merges[n1_in_current_fsa]

    else:
        n1_in_current_fsa = n1

    return n1_in_current_fsa


if __name__ == "__main__":
    
    main()
