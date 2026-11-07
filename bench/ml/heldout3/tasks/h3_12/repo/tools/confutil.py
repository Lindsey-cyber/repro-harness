import json


def load(path, overrides=()):
    cfg = json.load(open(path))
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = json.loads(val)
    return cfg
