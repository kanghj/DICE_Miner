import sys
from collections import defaultdict

from fsa.rule_utils import getNF
from violations.violation_checker import learn_from_traces_for_model
import main


def run():
    print(sys.argv)
    traces = main.read_traces(sys.argv[1], main.strip_descriptor)

    traces = [trace for trace in traces if "<IO-LEAK>" not in trace]

    pure_functions = main.read_pure_functions(sys.argv[2],
                                         main.strip_descriptor)  # there are functions that we statically analyse to be pure!

    vocab = set()
    event_vocab = set()
    can_be_disaled = set()
    ctor = None
    for trace in traces:
        for event in trace:

            if isinstance(event, str):
                # print('is str')
                event_str = event
                full_event_str = event
            else:
                # print('is tuple')
                if 'non-null' in event[0]:
                    print('event:', event)

                event_str = event[0]
                full_event_str = event[0] + ":" + event[1] if event[1] is not None else event[0]





            if "EXCEPTION" in event_str:

                can_be_disaled.add(event_str)
                continue
            vocab.add(event_str)
            event_vocab.add(full_event_str)
            if main.is_constructor(event_str):
                ctor = event_str

    always_enabled = set()
    if len(can_be_disaled) > 0:  # we only want to initialize always enabled if there are actually exceptional behaviour tracked in the traces
        always_enabled = vocab - set(can_be_disaled) - set(["<START>", "<END>", ctor])

        print('always_enabled', always_enabled)

    enabledness_model = learn_from_traces_for_model(traces, pure_functions)

    # debug

    # print("reverse lang")
    reverse_traces = [reversed(trace) for trace in traces]
    reverse_enabledness_model = learn_from_traces_for_model(reverse_traces, pure_functions)
    for context, enabledness in reverse_enabledness_model.items():
        p1_enabled_tokens = set()
        for token, prob in enabledness.items():
            if prob >= 0.99999 and "EXCEPTION" not in token and token not in always_enabled:
                p1_enabled_tokens.add(token)
        for p1_enabled_token in p1_enabled_tokens:
            print("LTL:AIP", context[0], p1_enabled_token)
            print("LTL:AP", context[0], p1_enabled_token)

    getNF(enabledness_model, traces, always_enabled)

    # init by having all pairs in always_preceded_candidates
    always_preceded_candidates = defaultdict(lambda: defaultdict(int))
    token_counts = defaultdict(int)

    print('always preceded')

    for trace_num, trace in enumerate(traces):
        # print('trace', trace_num, trace)

        i = 0
        trace = [event for event in reversed([event for event in trace])]

        token_appears_in_trace = set()
        for event in trace:
            i += 1

            if isinstance(event, str):
                event_str = event
            else:
                event_str = event[0] + (":" + event[1] if event[1] is not None else "")

            token_counts[event_str] += 1
            events_occuring = set()


            for j, event2 in enumerate(trace[i:]):
                if isinstance(event2, str):
                    event_str2 = event2
                else:
                    event_str2 = event2[0] + (":" + event2[1] if event2[1] is not None else "")

                events_occuring.add(event_str2)

            for event_str2 in events_occuring:
                always_preceded_candidates[event_str][event_str2] += 1


    for key1, values in always_preceded_candidates.items():
        for key2, count in values.items():
            # print('key1', key1, 'key2', key2, 'count', count, 'token count', token_counts[key1])
            if count >= token_counts[key1]:
                print('LTL:AP', key1, key2)

    always_followed_candidates = defaultdict(lambda: defaultdict(int))
    token_counts = defaultdict(int)
    for trace_num, trace in enumerate(traces):
        i = 0

        for event in trace:

            if isinstance(event, str):
                event_str = event
            else:
                event_str = event[0] + (":" + event[1] if event[1] is not None else "")

            token_counts[event_str] += 1

            events_occuring = set()
            for j, event2 in enumerate(trace[i + 1:]):
                if isinstance(event2, str):
                    event_str2 = event2
                else:
                    event_str2 = event2[0] + (":" + event2[1] if event2[1] is not None else "")

                events_occuring.add(event_str2)
            for event_str2 in events_occuring:
                always_followed_candidates[event_str][event_str2] += 1
            i += 1

    for key1, values in always_followed_candidates.items():
        for key2, count in values.items():
            # print('key1', key1, 'key2', key2, 'count', count, 'token count', token_counts[key1])
            if count >= token_counts[key1]:
                print('LTL:AF', key1, key2)

    print('writing to ', sys.argv[3])
    with open(sys.argv[3], 'w+') as outfile:
        for event in vocab:
            outfile.write(event)
            outfile.write('\n')

    if len(sys.argv) > 4:
        print('writing to ', sys.argv[4])
        with open(sys.argv[4], 'w+') as outfile:
            for event in event_vocab:
                outfile.write(event)
                outfile.write('\n')



if __name__ == "__main__":
    run()
