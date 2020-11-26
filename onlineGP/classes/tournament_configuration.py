class TournamentConfiguration:
    def __init__(self, fit_threshold: float, tournament_size: int, train: bool = False):
        self.fit_threshold = fit_threshold
        self.tournament_size = tournament_size
        self.train = train
