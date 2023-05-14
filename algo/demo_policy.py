class DemoPolicy:
    def __init__(self):
        self.x = {0: 0, 1: 3, 2: 3, 3: 3,
                  4: 0, 5: -1, 6: 0, 7: -1,
                  8: 3, 9: 1, 10: 0, 11: -1,
                  12: -1, 13: 2, 14: 2, 15: -1}

    def get_action(self, s):
        return self.x[s]
