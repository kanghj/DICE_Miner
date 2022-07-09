import functools
import sys, os, random

from graphviz import Digraph
from collections import deque, Counter, defaultdict

from copy import deepcopy


debug = False


def is_accepted(fsa, events, debug=False):
    if isinstance(events, str):
        raise Exception("WRONG type of events!")
    current_nodes = set(fsa.startings)
    # print(events)
    # print('type of transitions', [edge[2] for edge in fsa.transitions if edge[0] in current_nodes])
    for event in events:
        avail_edges = [edge for edge in fsa.transitions if edge[0] in current_nodes and edge[2] == event]

        if len(avail_edges) == 0:
            return False

        current_nodes = [edge[1] for edge in avail_edges]

    return True



def bfs(n, adjlist):
    q = deque()
    q.append(n)
    visited = set()
    visited.add(n)
    while len(q) > 0:
        n = q.popleft()
        if n not in adjlist:
            continue
        for v in adjlist[n]:
            if v not in visited:
                visited.add(v)
                q.append(v)
    return visited




class MutableEdge:
    def __init__(self, from_node, to_node, label):
        self.from_node = from_node
        self.to_node = to_node
        self.label = label

    # def __eq__(self, other):
    #     return self.from_node == other.from_node and self.to_node == other.to_node and self.label == other.label

    def __str__(self):
        return ','.join([str(self.from_node), str(self.to_node), self.label])


class StandardAutomata:
    def __init__(self, startings, edges, endings):
        self.transitions = list(edges)
        self.startings = set(startings)
        self.endings = set(endings)
        self.states = set()
        for p in edges:

            (source, dest, label) = p
            self.states.add(source)
            self.states.add(dest)

        self.states.update(set(startings))

        # meta data used to handle and optimize updates
        self.mappings_forward = defaultdict(list)
        self.mappings_back = defaultdict(list)
        self.removed_states = set()

        # suffix
        self.prefixes = defaultdict(list)
        self.suffixes = defaultdict(list)

    def size(self):
        return len(self.states) - len(self.removed_states)

    def vocab(self):
        vocab = set()
        for edge in self.transitions:
            vocab.add(edge[2])
        return vocab


    def serialize(self, f):
        import json
        json_string = json.dumps([list(self.transitions), list(self.startings), list(self.endings), list(self.states)])
        with open(f, 'w+') as opened:
            opened.write(json_string)

    @staticmethod
    def deserialize(f):
        import json

        with open(f) as infile:
            d = json.load(infile)
        edges = set()
        for edge in d[0]:
            edges.add(tuple(edge))

        return StandardAutomata(d[1], edges, d[2])

    def clone(self):
        new_fsa= StandardAutomata(self.startings.copy(), self.transitions.copy(), self.endings.copy())

        new_fsa.states = set()

        # new_fsa.removed_states = self.removed_states.copy()
        for s in self.states:
            if s not in self.removed_states:
                new_fsa.states.add(s)


        # new_fsa.mappings_forward = self.mappings_forward #deepcopy(self.mappings_forward)
        # new_fsa.mappings_back =  self.mappings_back #deepcopy(self.mappings_back)
        return new_fsa

    def find_delta(self, node):
        f = {}
        for (source, dest, label) in self.transitions:

            for v in node:
                if source != v:
                    continue
                if label not in f:
                    f[label] = set()
                f[label].add(dest)
        ans = {}
        for e in f.items():
            ans[e[0]] = tuple(sorted(list(e[1])))
        return ans

    def look_for_counter_example(self, node1, node2, preceding_event, event):
        befores = set()
        afters = set()
        for edges in self.mappings_forward.values():
            for edge in edges:
                if (edge.to_node == node2 or edge.to_node == node1):
                    befores.add(edge.from_node)
                if (edge.from_node == node2 or edge.from_node == node1):
                    afters.add(edge.to_node)
        visited = set()



    def nfa2dfa(self):
        print("Input states:", len(self.states))
        #########################################################################################
        q = deque()
        found_states = set()
        delta_function = {}
        for s in self.startings:
            q.append((s,))
            found_states.add((s,))
        while len(q) > 0:
            n = q.popleft()
            delta = self.find_delta(n)
            delta_function[n] = delta
            # print n, delta
            for x in delta.values():
                if x not in found_states:
                    q.append(x)
                    found_states.add(x)
        print("Found state:", len(found_states))
        ##########################################################################################
        starting_states = set([n for n in found_states for e in self.startings if e in n])
        ending_states = set([n for n in found_states for e in self.endings if e in n])
        dfa_transitions = set()
        for n in found_states:
            source = n
            for e in delta_function[n].items():
                label = e[0]
                dest = e[1]
                dfa_transitions.add((source, dest, label))
        ###########################################################################################

        state_index = sorted(list(found_states))
        state_mapping = {state_index[i]: str(i) for i in range(len(found_states))}

        g = StandardAutomata(list(map(lambda x: state_mapping[x], starting_states)),
                             list(map(lambda x: (state_mapping[x[0]], state_mapping[x[1]], x[2]), dfa_transitions)),
                             list(map(lambda x: state_mapping[x], ending_states)))
        return g

    def update(self, new_trace):
        assert len(self.startings) == 1
        current_nodes = self.startings

        for i, token in enumerate(new_trace):
            avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]

            if len(avail_edges) == 0:
                # nothing matches, need to extend FSA
                for curr_node in current_nodes:
                    new_node = len(self.states) + 1
                    self.states.add(new_node)
                    new_edge = (curr_node, new_node, token)
                    self.transitions.append(new_edge)

                    self.prefixes[curr_node].append(new_trace[:i])
                    self.suffixes[curr_node].append(new_trace[i+1:])



                avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]
                assert len(avail_edges) > 0

            # matches, update current_nodes by traversing all of these edges
            current_nodes = [edge[1] for edge in avail_edges]

    def update_self_loop_for_pure(self, new_trace, pures):
        assert len(self.startings) == 1
        current_nodes = self.startings

        for i, token in enumerate(new_trace):
            avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]

            if len(avail_edges) == 0:
                # nothing matches, need to extend FSA
                for curr_node in current_nodes:
                    without_value = token.split(':')[0]
                    # print('pures are')
                    # print(pures)
                    if without_value in pures:
                        # print('got pure?')
                        new_edge = (curr_node, curr_node, token)
                        self.transitions.append(new_edge)

                    else:
                        new_node = len(self.states) + 1
                        self.states.add(new_node)
                        new_edge = (curr_node, new_node, token)
                        self.transitions.append(new_edge)

                    self.suffixes[curr_node].append(new_trace[i+1:])

                avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]
                assert len(avail_edges) > 0

            # matches, update current_nodes by traversing all of these edges
            current_nodes = [edge[1] for edge in avail_edges]

    def update_only_add_self_loops(self, new_trace):
        """
        Like update, but do not create new states.
        :param traces:
        :return:
        """
        assert len(self.startings) == 1
        current_nodes = self.startings

        for token in new_trace:
            avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]

            if len(avail_edges) == 0:
                # nothing matches, need to consume this token/event by adding a self-loop
                for curr_node in current_nodes:
                    new_edge = (curr_node, curr_node, token)
                    self.transitions.append(new_edge)

                avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]
                assert len(avail_edges) > 0

            # matches, update current_nodes by traversing all of these edges
            current_nodes = [edge[1] for edge in avail_edges]


    def future_seq_splitting_pure_and_ordering(self, k, prefix, pures=None):
        """
        returns a map of node_index -> set of (sequences of impure method tokens, set of pure_tokens)
        :param k:
        :param prefix:
        :param pures:
        :return:
        """
        # traverse up to the prefix
        current_nodes = self.startings

        for token in prefix:

            avail_edges = [edge for edge in self.transitions if edge[0] in current_nodes and edge[2] == token]

            if len(avail_edges) == 0:
                raise Exception("prefix is rejected")

            # update current_nodes by traversing all of these edges
            current_nodes = [edge[1] for edge in avail_edges]

        # traverse all sequences of length k
        results = {} # a map of node_index -> list of sequences of (token, pures)s
        remaining_nodes = {}
        for current_node in current_nodes:
            results[current_node] = []

            remaining_nodes[current_node] = [(current_node, ([], set()))]

        # debug
        at_least_one_pure_detected = False
        for query_node in current_nodes:
            while len(remaining_nodes[query_node]) > 0:
                current_node, (seq, pure_funcs) = remaining_nodes[query_node].pop()
                # print('===')
                # print(current_node, seq, pure_funcs)

                next_pure_funcs = pure_funcs.copy()
                if len(seq) == k:
                    results[query_node].append((seq.copy(), next_pure_funcs))
                    continue

                avail_edges = [edge for edge in self.transitions if edge[0] == current_node]

                if len(avail_edges) == 0:
                    # can't traverse further
                    results[query_node].append((seq.copy(), next_pure_funcs))
                    continue

                next_labels = [edge[2] for edge in avail_edges]
                next_nodes = [edge[1] for edge in avail_edges]

                # add all the pure funcs
                for next_node, next_label in zip(next_nodes, next_labels):
                    next_label_without_value = next_label.split(':')[0]

                    # if next_label == "hasMoreTokens:FALSE":
                    #     print("HASMORETOKENS:FALSE", next_label_without_value in pures)
                    if next_label_without_value in pures:
                        at_least_one_pure_detected = True
                        next_pure_funcs.add(next_label)

                for next_node, next_label in zip(next_nodes, next_labels):
                    next_label_without_value = next_label.split(':')[0]

                    next_seq = seq.copy()

                    if next_label_without_value not in pures:
                        next_seq.append(next_label)
                    # print('next seq', next_seq)

                    remaining_nodes[query_node].append((next_node, (next_seq, next_pure_funcs)))

        # print(results)

        return results

    def assert_global_invariants(self):
        """
        after construction, these invariants must never be false
        :return:
        """
        assert len(self.startings) > 0

        for edge in self.transitions:
            assert len(edge) == 3

            if edge[0] not in self.states:
                print('MISSING', edge[0], edge)
            assert edge[0] in self.states
            if edge[1] not in self.states:
                print('MISSING', edge[1], edge)
            assert edge[1] in self.states

    def bfs(self, avoid_self_loops=True):
        """
        BFS through the automaton, producing sequences of tokens.
        Warning: Does not traverse self-loops by default. <- thus this function is not suitable for generating traces.
        :return:
        """

        start = [(node, []) for node in self.startings]
        visited = set()
        remaining = []
        remaining.extend(start)

        result = []
        while len(remaining) > 0:
            node, seq = remaining.pop()
            if len(seq) > 0:
                result.append(seq)

            avail_edges = [edge for edge in self.transitions
                           if edge[0] == node and (not avoid_self_loops or edge[0] != edge[1])]

            visited.add(node)

            next_nodes = [edge[1] for edge in avail_edges]
            next_labels = [edge[2] for edge in avail_edges]

            if len(seq) > 50: # prevent blowup of the number of possible states to consider
                continue

            remaining.extend([(next_node, (seq + [next_label])) for next_node, next_label in zip(next_nodes, next_labels)])

        return result


    def remove_unreachable_states_from_starting(self):
        # bfs to find reachable states
        adjlist = {}
        all_nodes = set()
        for e in self.transitions:
            source = e[0]
            dest = e[1]
            all_nodes.add(source)
            all_nodes.add(dest)
            if source not in adjlist:
                adjlist[source] = []
            adjlist[source] += [dest]
        # bfs
        visted = set()
        for x in self.startings:
            if x not in visted:
                found = bfs(x, adjlist)
                visted.update(found)

        # list unreachable states

        unreached_nodes = all_nodes - visted
        return self.remove_state(unreached_nodes)

    def __str__(self):
        return self.to_dot()

    def merge_states(self, state1, state2):
        def reverse(fsa, new_state_number, mutated_edges, old_values):
            """
            list of mutatble_edges, list of (node1, noed2, label)
            :param mutated_edges:
            :param old_values:
            :return:
            """
            fsa.states.remove(new_state_number)
            for mutable_edge, old_value in zip(reversed(mutated_edges), reversed(old_values)):
                # if debug is True:
                #     print('reversing', mutable_edge.from_node, mutable_edge.to_node, mutable_edge.label)
                #     print('back to ', old_value)
                mutable_edge.from_node = old_value[0]
                mutable_edge.to_node = old_value[1]
                mutable_edge.label = old_value[2]

                # print('back', mutable_edge.from_node, mutable_edge.to_node, mutable_edge.label)

                if old_value[0] in fsa.removed_states:
                    fsa.removed_states.remove(old_value[0])
                if old_value[1] in fsa.removed_states:
                    fsa.removed_states.remove(old_value[1])
            del fsa.mappings_forward[new_state_number]
            del fsa.mappings_back[new_state_number]


        self.done_merge = False

        # fsa_clone = self.clone()

        new_state_number = max(self.states) + 1
        self.states.add(new_state_number)
        self.removed_states.add(state1)
        self.removed_states.add(state2)

        # print('merge', state1, state2, '=>', new_state_number)
        if len(self.mappings_back) == 0 or len(self.mappings_forward) == 0:
            # init
            for i, edge in enumerate(self.transitions):
                node1 = edge[0]
                node2 = edge[1]
                mutable_edge = MutableEdge(node1, node2, edge[2])
                if mutable_edge not in self.mappings_forward[node1]:
                    self.mappings_forward[node1].append(mutable_edge)
                if mutable_edge not in self.mappings_back[node2]:
                    self.mappings_back[node2].append(mutable_edge)

        # all_edges = set()
        # for key, edges in self.mappings_back.items():
        #     all_edges.update(edges)
        # print('total edges', len(all_edges))


        for_reverse = []
        mutated = []

        edges_affected_1 = self.mappings_back[state1]
        edges_affected_2 = self.mappings_back[state2]
        edges_affected_3 = self.mappings_forward[state1]
        edges_affected_4 = self.mappings_forward[state2]

        for mutable_edge in edges_affected_1 + edges_affected_2:

            if mutable_edge.to_node in [state1, state2]:
                mutated.append(mutable_edge)
                for_reverse.append((mutable_edge.from_node, mutable_edge.to_node, mutable_edge.label))

                mutable_edge.to_node = new_state_number
            elif mutable_edge.to_node != new_state_number:
                print('merging', state1, state2, '->', new_state_number)
                print("WARN", 'expecting', mutable_edge.to_node, 'to be one of', [state1, state2])
                assert False

            # if mutable_edge.to_node == new_state_number and mutable_edge not in self.mappings_back[new_state_number]:
            self.mappings_back[new_state_number].append(mutable_edge)

        for mutable_edge in edges_affected_3 + edges_affected_4:
            if mutable_edge.from_node in [state1, state2]:
                mutated.append(mutable_edge)
                for_reverse.append((mutable_edge.from_node, mutable_edge.to_node, mutable_edge.label))

                mutable_edge.from_node = new_state_number
            elif mutable_edge.from_node != new_state_number    :
                print('merging', state1, state2, '->', new_state_number)
                print("WARN", 'expecting', mutable_edge.from_node, 'to be one of', [state1, state2])
                assert False

            # if mutable_edge.from_node == new_state_number and mutable_edge not in self.mappings_forward[new_state_number]:
            self.mappings_forward[new_state_number].append(mutable_edge)

        # print('number of edges (non-unique)', len(self.mappings_back[new_state_number]), len(self.mappings_forward[new_state_number]))

        if debug is True:
            # asserts that all edges have been updated
            for mutable_edges in self.mappings_back.values():
                for mutable_edge in mutable_edges:
                    if mutable_edge.from_node in [state1, state2]:
                        print('failed merging states ', [state1, state2], 'to new state:', new_state_number, '. Stray edge is', mutable_edge)
                        print('is it in mappings?', mutable_edge in self.mappings_forward[state1], mutable_edge in self.mappings_forward[state2])
                        print('is it in mappings?', mutable_edge in self.mappings_back[state1],
                              mutable_edge in self.mappings_back[state2])
                        print('what mappings is it in?')
                        for mappings_key in self.mappings_back.keys():
                            if mutable_edge in self.mappings_back[mappings_key]:
                                print(mappings_key)
                        for mappings_key in self.mappings_forward.keys():
                            if mutable_edge in self.mappings_forward[mappings_key]:
                                print(mappings_key)

                        assert mutable_edge.from_node not in [state1, state2]
            for mutable_edges in self.mappings_forward.values():
                for mutable_edge in mutable_edges:
                    if mutable_edge.from_node  in [state1, state2]:
                        print('edge is', mutable_edge)
                        assert mutable_edge.from_node not in [state1, state2]


        return functools.partial(reverse,  self, new_state_number, mutated, for_reverse), new_state_number

    def done_merging(self):
        if len(self.mappings_back) > 0:
            self.transitions = set()

        # assert len(self.mappings_back) == len(self.mappings_forward)
            for edge_set in self.mappings_back.values():
                edges = [(n.from_node, n.to_node, n.label) for n in edge_set
                         if n.from_node not in self.removed_states and n.to_node not in self.removed_states]
                self.transitions.update(edges)

        # self.states = set()
        # for p in edges:
        #     (source, dest, label) = p
        #     self.states.add(source)
        #     self.states.add(dest)
        #
        # self.states.update(set(self.startings))

        # remap numbers.
        old_state_ids_to_new_states = {}
        new_state_id_nums = 1

        new_states = set()
        for (node1, node2, label) in self.transitions:
            if "init>" in label: # state 0 should be the state after the constructor, for AFLNet usage.
                if 0 not in old_state_ids_to_new_states:
                    new_states.add(0)
                    old_state_ids_to_new_states[node2] = 0
            if "<START" in label:
                new_states.add(99999)
                old_state_ids_to_new_states[node1] = 99999
                new_states.add(999998)
                old_state_ids_to_new_states[node2] = 999998

            if node1 not in old_state_ids_to_new_states:
                new_states.add(new_state_id_nums)
                old_state_ids_to_new_states[node1] = new_state_id_nums

                print('assign ', node1, 'to', new_state_id_nums)
                new_state_id_nums += 1
            if node2 not in old_state_ids_to_new_states:
                new_states.add(new_state_id_nums)
                old_state_ids_to_new_states[node2] = new_state_id_nums

                print('assign ', node2, 'to', new_state_id_nums)
                new_state_id_nums += 1

        self.states = new_states
        # print("states")
        # print(self.states)

        new_transitions = []
        for edge in self.transitions:
            new_transitions.append((old_state_ids_to_new_states[edge[0]], old_state_ids_to_new_states[edge[1]], edge[2]))

        self.transitions = new_transitions

        # shift the startings and endings around to make compatible with the ground-truth FSAs.
        self.startings.clear()
        self.endings.clear()

        for edge in self.transitions:
            if edge[2] == "<START>":
                self.startings.add(edge[1])
            if edge[2] == "<END>":
            # actually, any state is an ending state, as a client of any library can stop executing at any moment...
            # else:
                self.endings.add(edge[1])

        self.done_merge = True





    def remove_state(self, to_remove_states):
        print("Removing the following states:", to_remove_states)

        startings = set(self.startings)
        endings = set(self.endings)
        edges = set()
        for x in to_remove_states:
            if x in startings:
                startings.remove(x)
            if x in endings:
                endings.remove(x)

        for e in self.transitions:
            source = e[0]
            dest = e[1]
            if source in to_remove_states or dest in to_remove_states:
                continue
            edges.add(e)
        return StandardAutomata(startings, edges, endings)

    def find_nodes_connected_by(self, label):
        results = set()
        for edges in self.mappings_forward.values():
            for edge in edges:
                if edge.label == label:
                    results.add((edge.to_node, edge.from_node))

        return results



    def is_immediately_followed(self, label1, label2, pure_funcs):

        # print('label1', label1)
        label1_nodes = self.find_nodes_connected_by(label1)
        # print('label1-node',label1_nodes)

        # input()
        remaining_nodes = []
        # print([(to_node, []) for from_node, to_node in label1_nodes])
        remaining_nodes.extend([(to_node, [], 0) for from_node, to_node in label1_nodes])
        while len(remaining_nodes) > 0:
            node, seq, self_loops = remaining_nodes.pop()
            # print(node, seq)
            if self_loops >= 1:
                continue
            if len(seq) > 10:
                continue
            edges = self.mappings_forward[node]

            for edge in edges:
                # print(edge[2], label2)
                if edge.label == label2:
                    # FOUND
                    print('FOUND')
                    return True

                if edge.label not in pure_funcs:
                    # ignore impure functions when traversing further
                    continue
                if edge[0] == edge[1]:
                    remaining_nodes.append((edge[1], seq + [edge[2]], self_loops + 1))
                else:
                    remaining_nodes.append((edge[1], seq + [edge[2]], 0))

        return False

    def is_followed(self, label1, label2, pure_funcs, state_a= None, prefix = None, node_suffixes=None):

        # print('label1', label1)
        if state_a is None:
            label1_nodes = self.find_nodes_connected_by(label1)
        else:
            label1_nodes = self.mappings_forward[state_a]
        # print('label1-node',label1_nodes)

        # assume that state_a is prefixed by label1 at some point
        has_label_in_suffix = False
        for suffix in node_suffixes:
            if label2 in suffix:
                has_label_in_suffix = True
                # print('\tviolated at prefix=', prefix,'suffix=', suffix, ' by NF', label1, label2)
                return True

        return False

    def is_always_followed(self, label1, label2, pure_funcs):

        label1_nodes = self.find_nodes_connected_by(label1)

        remaining_nodes = []
        remaining_nodes.extend([(to_node, [], 0) for from_node, to_node in label1_nodes])
        while len(remaining_nodes) > 0:
            node, seq, self_loops = remaining_nodes.pop()
            # print(node, seq, self_loops)
            # print(node, seq)
            if self_loops >= 1:
                continue
            if len(seq) > 15:
                continue
            edges = self.mappings_forward[node]
            if len(edges) == 0:
                # no edges forward? uh no, this means the rule is violated
                return False

            for edge in edges:
                if edge.label == label2:
                    # objective met
                    continue

                if edge.label not in pure_funcs:
                    # ignore impure functions when traversing further
                    continue

                # self-loop,
                if edge.from_node == edge.to_node:
                    # increase self-loop count. Need to track this to prevent infinite loop
                    remaining_nodes.append((edge.to_node, seq + [edge.label], self_loops + 1))
                else:
                    remaining_nodes.append((edge.to_node, seq + [edge.label], 0))

        return True

    def is_preceded(self, label1, label2, pure_funcs):
        label2_nodes = self.find_nodes_connected_by(label2)

        remaining_nodes = []
        remaining_nodes.extend([(from_node, [], 0) for from_node, to_node in label2_nodes])
        while len(remaining_nodes) > 0:
            node, seq, self_loops = remaining_nodes.pop()

            if self_loops >= 1:
                continue
            if len(seq) > 5:
                continue
            edges = self.mappings_back[node]

            for edge in edges:
                if edge.label == label1:
                    # print('FOUND')
                    return True

                if edge.label not in pure_funcs:
                    continue

                # self-loop,
                if edge.from_node == edge.to_node:
                    # increase self-loop count. Need to track this to prevent infinite loop
                    remaining_nodes.append((edge.to_node, seq + [edge.label], self_loops + 1))
                else:
                    remaining_nodes.append((edge.to_node, seq + [edge.label], 0))

        return False

    def is_purely_preceded(self, label1, label2):
        pass

    def is_state_have_disabled_methods(self, state, disabled_methods):

        forwards = self.mappings_forward[state]
        result = None
        for edge in forwards:
            if edge.label in disabled_methods:
                result = True

        return result

    def is_state_have_disabled_methods_backwards(self, state, disabled_methods):

        forwards = self.mappings_back[state]
        result = None
        for edge in forwards:
            if edge.label in disabled_methods:
                result = True
                if debug is False:
                    break
        return result


    def to_dot(self):

        f = Digraph('Automata', format='eps')
        f.body.extend(['rankdir=LR', 'size="8,5"'])
        f.attr('node', shape='star')
        for n in self.startings:
            f.node(str(n))
        f.attr('node', shape='doublecircle')
        for n in self.endings:
            f.node(str(n))
        f.attr('node', shape='circle')
        for (source, dest, label) in self.transitions:
            f.edge(str(source), str(dest), label=label)

        return f.source

    def to_evaluation_format(self):
        outlines = [len(self.startings)]
        for n in self.startings:
            outlines += [n]
        outlines += [len(self.endings)]
        for n in self.endings:
            outlines += [n]
        outlines += [str(len(self.transitions))]
        for e in self.transitions:
            outlines += [str(e[0]) + '\t' + str(e[1]) + '\t' + (e[2])]
        return '\n'.join([str(x) for x in outlines])

    def is_accepting_one_trace(self, tr, adjlst, debug=False):
        try:
            rejected_prefixes = []
            for s in self.startings:

                flag, rejected_prefix = is_accepted_bfs(s, tr, adjlst, debug=debug)
                if flag:
                    return flag, None
                else:
                    rejected_prefixes += [rejected_prefix]

            return False, rejected_prefixes
        except Exception as e:
            print(e)
            return False, None
        finally:
            pass

    def create_adjacent_list(self):
        ans = {n: [] for n in self.states}
        for (source, dest, label) in self.transitions:
            ans[source] += [(dest, label)]
        for n in self.states:
            ans[n] = sorted(list(set(ans[n])))
        return ans


def is_accepted_bfs(init_node, tr, adjlst, debug=False):
    current_nodes = set([init_node])

    for current_index in range(len(tr)):
        current_label = tr[current_index]
        next_nodes = set()
        for source in current_nodes:
            for (dest, label) in adjlst[source]:
                if label != current_label:
                    continue

                next_nodes.add(dest)
        if len(next_nodes) == 0 and current_index < len(tr) - 1:
            return False, (tr[:current_index + 1], init_node)
        current_nodes = next_nodes

    return True, None
