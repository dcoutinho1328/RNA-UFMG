import numpy as np

# Generic function to read file content
def readFromFile(path):

    content = []

    with open(path, 'r') as file:
        for line in file.readlines()[1:]:
            content.append([])
            data = line.split()[1:]
            for d in data:
                content[-1].append(float(d))
    
    return np.matrix(content)