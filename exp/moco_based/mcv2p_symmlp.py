from exp.mocov2_plus import Trainer as MoCov2_Plus_Trainer


class Trainer(MoCov2_Plus_Trainer):
    def __init__(self):
        super(Trainer, self).__init__()

        self.MLP = 'mcv2p'
