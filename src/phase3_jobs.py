"""Emit the phase-3 job list: cutoff model kind pair-word1 pair-word2 (d28/d56 first)."""
MODS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]; CY = ["AAA", "BBB", "DDD", "EEE", "FFF", "GGG"]
for c in [28, 56, 7, 14, 112]:
    for m in ["XGB", "LR"]:
        for mo in MODS: print(c, m, "real", "LOMO", mo)
        for mo in CY: print(c, m, "real", mo, "13->14")
        for mo in MODS: print(c, m, "null", "NULL", mo)
