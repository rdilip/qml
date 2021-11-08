import numpy as np

class DataTracker:
    def __init__(self):
        self.data = {}
        self.data_vals = {}

    def register(self, label, val):
        self.data[label] = []
        self.data[label].append(val)
        self.data_vals[label] = val

    def update(self):
        for label in self.data:
            self.data[label].append(self.data_vals[label])

if __name__ == '__main__':
    A = DataTracker()
    train = np.random.rand()
    A.register("train", train)

    for i in range(12):
        train = np.random.rand()
        print(train)
        A.update()
    print(A.data)

