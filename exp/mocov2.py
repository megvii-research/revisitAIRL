from exp.mocov1 import Trainer as MoCo_v1_Trainer


class Trainer(MoCo_v1_Trainer):
    def __init__(self):
        super(Trainer, self).__init__()

        self.scheduler = 'warmcos'
        self.temperature = 0.2

        self.aug_plus = True

        self.MLP = 'moco'