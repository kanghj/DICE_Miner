from collections import defaultdict

def learn_from_traces_for_model(traces, pures):
    model_counts = {}
    context_counts = {}
    context_len = 1

    vocab = set()
    for trace in traces:
        context = []
        pure_events = set()

        for event in trace:

            assert len(context) <= context_len

            if isinstance(event, str):
                event_str = event
            else:
                event_str = event[0] + (":" + event[1] if event[1] is not None else "")

            vocab.add(event_str)

            if len(context) == context_len:
                add_to_context_and_model_counts(context, context_counts, event_str, model_counts)
                for pure_event in pure_events:
                    # print("adding pure, event", pure_event, event_str, "of context", context)
                    add_to_context_and_model_counts((pure_event,), context_counts, event_str, model_counts)

            event_without_retval = event[0]

            if event_without_retval in pures:
                # pure functions get excluded from building up the past context, but each are part of the context
                pure_events.add(event_str)
                continue

            context.append(event_str)
            if len(context) > context_len:
                context.pop(0)
                for pure_event in pure_events:
                    for pure_event_2 in pure_events:
                        add_to_context_and_model_counts((pure_event,), context_counts, pure_event_2, model_counts)
                pure_events.clear()

    model = {}
    for context, token_counts in model_counts.items():
        context_count = context_counts[context]
        model[context] = {}
        for event in vocab:
            # initialize
            model[context][event] = -1

        for token, counts in token_counts.items():
            model[context][token] = float(counts) / context_count

    if context_len == 1:
        for word in vocab:
            if (word,) not in model:
                model[(word,)] = {}
                for event in vocab:
                    # initialize
                    model[(word,)][event] = -1

    return model


def add_to_context_and_model_counts(context, context_counts, event_str, model_counts):

    if tuple(context) not in context_counts:
        context_counts[tuple(context)] = 0
    context_counts[tuple(context)] += 1
    if tuple(context) not in model_counts:
        model_counts[tuple(context)] = defaultdict(int)
    model_counts[tuple(context)][event_str] += 1


def inferred_rules(forward_model, backwards_model):
    """
    Print temporal constraints implied from the language model
    :return:
    """
    rules = []
    violations = set()
    for context, token_prob in forward_model.items():
        for token, prob in token_prob.items():
            if prob < 0:
                rule = ("NIF", context, token)
                rules.append(rule)
                rule = ("NF", context, token)
                rules.append(rule)
            else:
                # check if this transition violates any existing rules
                for rule in rules:
                    if rule[0] == "NF" or rule[0] == "NIF":
                        if rule[1] == context and rule[2] == token:
                            # violation!
                            violations.add(("NF", context, token))
    for violation in violations:
        if violation in rules:
            rules.remove(violation)


    # for context, token_prob in backwards_model.items():
    #     for token, prob in token_prob.items():
    #         if prob >= 0.99:
    #             rule = ("AP", context, token)
    #             rules.append(rule)
    #

    remaining_nodes = []

    #
    return rules


if __name__ == "__main__":
    def split_return_value(token):
        if ':' in token:
            return token.split(':')
        else:
            return token, None

    inputs = [["<START>", "ArrayList", "isEmpty:TRUE", "indexOf", "addAll:TRUE"]]
    processed_input = []
    for trace in inputs:
        processed_input_row = []
        processed_input.append(processed_input_row)
        for event in trace:
            processed_input_row.append(split_return_value(event))
    learn_from_traces_for_model(processed_input,
                                set(["toArray", "size", "isEmpty", "indexOf", "listIterator", "get"]))