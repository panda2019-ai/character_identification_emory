class CorefParams:
    nb_fltrs = 280
    nb_epoch = 50
    batch_size = 128
    gpu = []
    eval_only = False


class LinkingParams:
    nb_fltrs = 280
    nb_epoch = 100
    batch_size = 128
    gpu = []
    # 设置为True时，直接加载模型而不训练
    eval_only = True
