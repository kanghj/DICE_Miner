
def read_traces(path, strip_descriptor=None, get_test_id=False):
    if strip_descriptor is None:
        raise Exception("please pass descriptor config!")
    def split_return_value(token):
        if ':' in token:

            return tuple(token.split(':'))
        else:
            return (token, None)

    def is_exception(token):
        if token.startswith("EXCEPTION"):
            return True
        else:
            return False

    def strip_descriptor_info(token):

        assert isinstance(token, tuple)
        assert len(token) == 2

        if '(' in token[0]:
            assert len(token[0].split('(')[0]) > 0
            return (token[0].split('(')[0], token[1])
        else:
            assert len(token) > 0
            return token


    def split_trace_if_exceptional(tokens):
        exceptional_traces = []
        exceptional_traces.append([token for token in tokens if not is_exception(token)])  # the original trace

        built_trace = []
        for token in tokens:
            if not is_exception(token):
                built_trace.append(token)
            else:
                exceptional_trace = list(built_trace)
                # exceptional_trace.append(token)
                exceptional_trace.append("<END>")
                exceptional_traces.append(exceptional_trace)
                break

        return exceptional_traces


    results = []

    test_id_mappings = {}
    already_seen = set()
    with open(path) as infile:
        lines = []
        for i, line in enumerate(infile):


            if get_test_id:  # in some versions, i use "//" comments for debugging the test id
                test_id = line.split("//")[1] if len(line.split("//")) > 1 else None
                if test_id is not None:
                    test_id_mappings[i] = test_id.strip()

            line = line.split("//")[0] # in some versions, i use "//" comments for debugging the test id

            if line in already_seen:
                continue
            already_seen.add(line)
            lines.append(line)

        for line in sorted(lines):
            splitted = line.rstrip().split()

            if splitted[0] != "<START>":
                splitted.insert(0, "<START>")
            if splitted[-1] != "<END>":
                splitted.append("<END>")

            traces = split_trace_if_exceptional(splitted)

            for trace in traces:
                results.append([strip_descriptor_info(split_return_value(token)) for token in trace])

    if get_test_id:
        return results, test_id_mappings
    return results
#
# if __name__ == "__main__":
#     print(read_traces("/Users/kanghongjin/evosuite_learning/Tutorial_Maven/.evosuite/tmp_2019_08_11_14_11_51/tests/tutorial.util.LinkedList\$ListItr.traces", 400, main.strip_descriptor))
