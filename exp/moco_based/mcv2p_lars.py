from exp.mocov2_plus import Trainer as MoCov2Plus_Trainer


class Trainer(MoCov2Plus_Trainer):
    def __init__(self):
        super(Trainer, self).__init__()

        self.lars = True

        self.bn = 'plainsync'
        self.bn_target = 'plainsync'

        self.lr = 0.3
        self.weight_decay = 1e-6
        self.weight_decay_exclude = 0

        self.lr_cls = 0.2
        self.scheduler_cls = 'cos'

    def build_optimizer(self, args):
        if args.total_epochs == 100:
            self.lr = 0.45
        elif args.total_epochs == 200 or args.total_epochs == 300:
            self.lr = 0.3
        else:
            raise ValueError('The number of total training epochs is unrecognized.')

        super(Trainer, self).build_optimizer(args)