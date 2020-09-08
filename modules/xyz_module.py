#A tiny module to write itterables to .XYZ files

def savexyz(data, filename):
    if filename[-4:].lower() != ".xyz":
        filename += ".xyz"
    with open(filename, "w") as outfile:
        for i in data:
            outfile.write(f"{len(i)}\n")
            outfile.write("\n")
            for j in i:
                outfile.write("H")
                for k in j:
                    outfile.write(f"\t{k}")
                outfile.write("\n")
"""
def loadxyz(filename):
    data = []
    if filename[-4:].lower() != ".xyz":
        filename += ".xyz"
    with open(filename, "r") as infile:
        
        r = np.zeros((,3))
"""