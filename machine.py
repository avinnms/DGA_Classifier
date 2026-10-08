import math
from collections import Counter

def entropy(s):
    count = Counter(s)
    return -sum(c/len(s) * math.log2(c/len(s)) for c in count.values())


