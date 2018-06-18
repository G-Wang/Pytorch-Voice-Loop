class Hparams:
    """Experiment hyperparameters

    """
    def __init__(self):
        self.k = 20 # length of memory buffer
        self.dp = 100 # embedding output dim for sentece/input, default 256 from paper implementation
        self.do = 63 # output/vocoder feature dim, default 256 from paper implementation
        self.ds = 100
        self.c = 10 # number of mixtures in GMM attenion
        self.si = 44 # setence phoneme vocab size
        self.d = self.dp + self.do # buffer dimension
        self.ns = 21 # number of speakers
        self.attention_alignment = 0.05 # attention alignment
        self.noise_range = 4.0 # additive noise for teacher forcing

