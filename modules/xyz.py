#A tiny module to write itterables to .XYZ files
import os

class file:
    def __init__(self, fname, overwrite=False):
        if fname[-4:].lower() != ".xyz":
            fname += ".xyz"
        if not overwrite:
            if os.wpath.exists(fname):
                print("File already exists and overwrite is disabled.")
                print("Either choose a different file name / path, or set overwrite=True.")
                quit()
        else:
            if os.wpath.exists(fname):
                print(f"Overwriting file {fname}!")
            self.file = open(fname, "w")

    def save(self, data):
        with self.file as outfile:
        for i in data:
            outfile.write(f"{len(i)}\n")
            outfile.write("\n")
            for j in i:
                outfile.write("H")
                for k in j:
                    outfile.write(f"\t{k}")
                outfile.write("\n")

"""
def load(filename):
    data = []
    if filename[-4:].lower() != ".xyz":
        filename += ".xyz"
    with open(filename, "r") as infile:
        
        r = np.zeros((,3))
"""