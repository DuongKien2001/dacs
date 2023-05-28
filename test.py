"""
from tensorboardX import SummaryWriter

summary_writer = SummaryWriter(log_dir='')

f = open("log_ema_update_prototype_1.log", "r")
lines = f.readlines()
for i in range(len(lines)):
    l = lines[i].split()
    if len(l) == 5:
        if l[2] == "Training/contrastive_feat_loss":
            summary_writer.add_scalar('Train/contrastive_feat_loss', float(l[3]), l[4])
        if l[2] == "Training/contrastive_out_loss":
            summary_writer.add_scalar('Train/contrastive_out_loss', float(l[3]), l[4])
    if len(l) == 6:
        if l[2] == "Training/Supervised":
            summary_writer.add_scalar('Train/loss_supervised', float(l[4]), l[5])
        if l[2] == "Training/Unsupervised":
            summary_writer.add_scalar('Train/loss_unsupervised', float(l[4]), l[5])

f = open("log_ema_update_prototype.txt", "r")
lines = f.readlines()
for i in range(len(lines)):
    l = lines[i].split()
    if len(l) == 7 and l[2] != 'Saving':
        summary_writer.add_scalar('Train/contrastive_feat_loss', float(l[3]), l[6])
        summary_writer.add_scalar('Train/contrastive_out_loss', float(l[5]), l[6])
    if len(l) == 9:
        summary_writer.add_scalar('Train/loss_supervised', float(l[4]), l[8])
        summary_writer.add_scalar('Train/loss_unsupervised', float(l[7]), l[8])
"""
from core.configs import cfg
from core.utils.prototype_dist_estimator import prototype_dist_estimator
from matplotlib import pyplot as plt
import numpy as np
from umap import UMAP

feat_estimator = prototype_dist_estimator(feature_num=2048, cfg=cfg)
if True:
    out_estimator = prototype_dist_estimator(feature_num=19, cfg=cfg)
j = feat_estimator.Proto.cpu()
umap2d = UMAP(init='random', random_state=0)
proj_2d = umap2d.fit_transform(j)
classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider",
        "car", "truck", "bus",
        "train", "motorcycle", "bicycle"))
plt.scatter(proj_2d[0], proj_2d[1])
for i in range(19):
    plt.text(proj_2d[i][0], proj_2d[i][1], classes[i])
plt.show()

