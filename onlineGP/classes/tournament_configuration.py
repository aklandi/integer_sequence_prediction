class TournamentConfiguration:
    def __init__(self, fit_threshold: float, tournament_size: int, pretrain: bool = True):
        self.fit_threshold = fit_threshold
        self.tournament_size = tournament_size
        self.pretrain = pretrain
