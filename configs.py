# --------------------------------------------------------
# 网络中可以修改的参数都需要在此文件中修改
# --------------------------------------------------------

# 训练图片存放位置（推荐使用绝对路径）
TRAIN_IMG_DATA_PATH = './experiments/experiment1/data/images'

# 训练图片对应标注文件存放位置（推荐使用绝对路径）
LABEL_PATH = './experiments/experiment1/data/xmls'

# 数据集中所包含的种类
CLASSES = ('__background__', 'AllowUTurn')

# 训练迭代轮数
MAX_EPOCH = 10

# 图片归一化后的最大宽度与高度
RESIZED_IMAGE_SIZE = [640, 960]

# 训练集中较难识别的目标是否参与训练
USE_DIFFICULT = False

# anchor的放大倍数，原始尺寸为16*16
ANCHOR_SCALES = [8, 16, 32]

# anchor的长宽比
ANCHOR_RATIOS = [0.5, 1.0, 2.0]

# 分类阶段输入的rois数目
BATCH_SIZE = 128

#非背景比例
FG_FRACTION = 0.25

# 训练时，若当前ROI对应的概率大于0.5，认为它是前景
FG_THRESH = 0.5

