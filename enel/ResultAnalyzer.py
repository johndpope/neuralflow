import pickle


class ResultAnalyzer:
    def __init__(self, pickle_file: str):
        self.__data = pickle.load(open(pickle_file, "rb"))

    def get_best_row(self, key):
        sorted_rows = sorted(self.__data, key=lambda x: x[key])
        return sorted_rows[0]


if __name__ == "__main__":
    file = "/home/giulio/tensorBoards/enel/24_24_single/results.pkl"

    r = ResultAnalyzer(file)
    # print(r.get_best_row("MAE_Ts"))
    print(r.get_best_row("MAE_Val"))

