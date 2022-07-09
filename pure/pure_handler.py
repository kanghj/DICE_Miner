

def read_pure_functions(path, strip_descriptor):
    # path = "./fsa/new-pure-methods.csv"
    results = []
    with open(path) as infile:
        for line in infile:
            value = line.rstrip()
            if strip_descriptor:
                value = value.split('(')[0]

            results.append(value)
    print('reading pure methods', results)
    return results


