import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run]= result

    def reset(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def print_statistics(self, run=None):
            result =  torch.tensor(self.results)
            best_result = torch.tensor(result)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.4f} ± {r.std():.4f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.4f} ± {r.std():.4f}')

class Logger_Channel(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run]= result

    def reset(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def print_statistics(self, run=None):
            result =  torch.tensor(self.results)
            best_result = torch.tensor(result)

            for channel in range(19):
                r = best_result[:, channel]
                print(f'Channel {channel}: {r.mean():.4f} ± {r.std():.4f}')

    def return_data(self, run=None):
        result =  torch.tensor(self.results)
        best_result = torch.tensor(result)
        mean = []
        std = []
        for channel in range(19):
            r = best_result[:, channel]
            mean_str = f"{r.mean():.4f}"
            std_str = f"{r.std():.4f}"
            mean.append(float(mean_str))
            std.append(float(std_str))
        return mean, std
