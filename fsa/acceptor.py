from collections import defaultdict

from fsa.automata import StandardAutomata
from violations.violation_checker import learn_from_traces_for_model
from traces.trace_reader import read_traces


def infer_model(traces, pures, enabledness_model):
    startings = set()
    endings = set()
    edges = set()

    # initialize an empty automata
    startings.add(0)
    fsa = StandardAutomata(startings, edges, endings)

    # update the automata to accept each trace
    for trace in traces:
        if isinstance(trace[0], tuple):
            trace = [event + (":" + suffix if suffix is not None else "" )for event, suffix in trace]

        fsa.update(trace)
    assert len(fsa.states) > 0
    assert len(fsa.transitions) > 0
    assert len(fsa.startings) > 0

    context_item_disabledness = defaultdict(set)
    for context, item_probs in enabledness_model.items():

        for item, prob in item_probs.items():
            if prob < 0:
                context_item_disabledness[context[0]].add(item)

    remaining_nodes = []
    for starting in fsa.startings:
        remaining_nodes.append((starting, None, None, []))

    pure_functions_of_node = defaultdict(set)
    impure_functions_of_node = defaultdict(set)

    removed = set()
    while len(remaining_nodes) > 0:

        current_node, last_impure, starting_edge, current_pures = remaining_nodes.pop()
        # print(current_node, last_impure, starting_edge, current_pures )

        avail_edges = [edge for edge in fsa.transitions if edge[0] == current_node]
        if len(avail_edges) == 0:

            continue

        next_labels = [edge[2] for edge in avail_edges]
        next_nodes = [edge[1] for edge in avail_edges]
        for next_node, next_label in zip(next_nodes, next_labels):
            next_pures = current_pures.copy()
            without_value = next_label.split(':')[0]

            changed_current_node = None
            changed_last_impure = None
            # detect inconsistency
            for p in pure_functions_of_node[last_impure] | impure_functions_of_node[last_impure]:
                if p in context_item_disabledness[next_label] or next_label in context_item_disabledness[p]:

                    new_node = len(fsa.states) + 1
                    fsa.states.add(new_node)

                    new_transitions = []
                    for edge in fsa.transitions:
                        if edge[1] == last_impure:
                            new_edge = (edge[0], new_node, edge[2])
                            new_transitions.append(new_edge)
                    fsa.transitions.extend(new_transitions)

                    if starting_edge is not None:
                        if starting_edge not in removed:
                            fsa.transitions.remove(starting_edge)
                            removed.add(starting_edge)
                        fsa.transitions.append((new_node, starting_edge[1], starting_edge[2]))
                    else:
                        if (current_node, next_node, next_label) not in removed:
                            fsa.transitions.remove((current_node, next_node, next_label))
                            removed.add((current_node, next_node, next_label))
                        fsa.transitions.append((new_node, next_node, next_label))

                        # in this case, the edge to be removed is the current edge
                        # current_node = new_node
                        changed_current_node = new_node

                        changed_last_impure = new_node
                    pure_functions_of_node[last_impure].update(current_pures)
                    # once the current edge is discovered to be inconsistent, it starts a new fork and we don't need to worry about further inconsistency with this edge
                    break

            if without_value not in pures:  # impure
                remaining_nodes.append((next_node, next_node, None, []))
                if changed_last_impure is None:
                    impure_functions_of_node[last_impure].add(next_label)
                else:
                    impure_functions_of_node[changed_last_impure].add(next_label)

            else:

                if changed_last_impure is None:
                    pure_functions_of_node[last_impure].add(next_label)
                else:
                    pure_functions_of_node[changed_last_impure].add(next_label)

                next_pures.append(next_label)
                if changed_current_node is None:
                    remaining_nodes.append((next_node, last_impure, (current_node, next_node, next_label), next_pures))
                else:
                    assert (changed_current_node, next_node, next_label) in fsa.transitions
                    remaining_nodes.append((next_node, last_impure, (changed_current_node, next_node, next_label), next_pures))

    return fsa

if __name__=="__main__":
    traces = [
        "<START> StringTokenizer hasMoreTokens:TRUE nextToken nextToken hasMoreTokens:FALSE <END>".split(),
        "<START> StringTokenizer hasMoreTokens:TRUE nextToken nextToken hasMoreTokens:TRUE <END>".split(),
        "<START> StringTokenizer hasMoreTokens:TRUE nextToken hasMoreTokens:FALSE <END>".split(),
        "<START> StringTokenizer hasMoreTokens:TRUE nextToken hasMoreTokens:TRUE nextToken nextToken hasMoreTokens:FALSE hasMoreTokens:FALSE <END>".split(),
        "<START> StringTokenizer hasMoreTokens:TRUE nextToken nextToken <END>".split(),
        "<START> StringTokenizer hasMoreTokens:TRUE nextToken nextToken nextToken <END>".split(),


        # "<START> StringTokenizer hasMoreTokens:TRUE nextToken countTokens hasMoreTokens:TRUE <END>".split(),
        # "<START> StringTokenizer hasMoreTokens:TRUE nextToken hasMoreTokens:TRUE nextToken nextToken hasMoreTokens:TRUE hasMoreTokens:TRUE <END>".split(),

    ]

    pure_functions = set(["hasMoreTokens", "countTokens"])
    traces_real = read_traces("/Users/kanghongjin/repos/DSM/data/StringTokenizer/input_traces/input.txt", 500)
    enabledness_model = learn_from_traces_for_model(traces_real, pure_functions)

    fsa = infer_model(traces, pure_functions, enabledness_model)

