class AverageMeter(object):
    # Computes and stores the average and current value
    """
       batch_time = AverageMeter()
       即 self = batch_time
       则 batch_time 具有__init__，reset，update三个属性，
       直接使用batch_time.update()调用
       功能为：batch_time.update(time.time() - end)
               仅一个参数，则直接保存参数值
        对应定义：def update(self, val, n=1)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        这些有两个参数则求参数val的均值，保存在avg中##不确定##

    """
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count