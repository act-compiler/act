def cost(path: str):
    with open(path, 'r') as f:
        return len(f.readlines())
