for c in [10, 25, 33, 50]:
    for m in ["XGB", "LR"]:
        for k in range(5):
            for kind in ["real", "null"]: print(c, m, k, kind)
