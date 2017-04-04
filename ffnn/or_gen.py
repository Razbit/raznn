from __future__ import print_function
import random
import sys

try:
    i = int(sys.argv[1])
    n = int (sys.argv[2])
    f_name = str(sys.argv[3])
except:
    print("Usage: or_gen <lines> <inputs> <filename>")
try:
    f = open(f_name, 'w')
    f.seek(0)
except:
    print("Cannot open file")
    
# Generate i lines
while i > 0:
    inputs = []
    for j in range (0, n):
        inputs.append(random.randint(0, 1))
    result = 0;
    for j in range (0, n):
        result = result or inputs[j]
    inputs.append(result)
    print(*inputs, sep=',', file=f)
    i = i - 1
f.truncate()
f.close()
