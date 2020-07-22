from cuhk03.model import *
from cuhk03.utils import *
from cuhk03.data import *
from reid.utils import feature_operate as FO
from resnet_ibn_b import *
from reid.utils.model_save_restore import *
from reid.data.samplers import RandomIdentitySampler
from reid.evaluation import market_evaluate
from cuhk03.data import *
import numpy as np
from cuhk03.model import ResNet

excel_path = '/content/drive/My Drive/Colab/ReID works/CVPR fintuning/excel/cuhk03/result_stn_cuhk03_9_3_32_172.csv'
weight_path = "/content/drive/My Drive/Colab/ReID works/CVPR fintuning/net_149.pth"

with open(weight_path, "rb") as f, open('./net_149.pth', 'wb') as fw:
    fw.write(f.read())

num_classes = 172

save_path = './'
train_transform = transforms.Compose([
    transforms.Resize((384, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((384, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
new_dataset = NewDataset(image_path=excel_path, transform=train_transform, use_onehot=False)
sampler = RandomIdentitySampler(new_dataset, num_instances=4)
train_loader = DataLoader(dataset=new_dataset, sampler=sampler, batch_size=32, num_workers=4)

query_info, gallery_info = get_query_gallery_info()
query_loader = get_loader(query_info)
gallery_loader = get_loader(gallery_info)

net = ResNet(num_classes=4101, num_features=1024)
net = restore_network("./", 149, net)
net = MNet(net, num_classes=num_classes, num_features=1024).cuda()
print("权重加载成功")

triplet = TripletLoss(0.3)


def loss(feat, score, feat_stn, score_stn, target):
    return F.cross_entropy(score, target) + F.cross_entropy(score_stn, target) + triplet(feat, target)[0] + \
           triplet(feat_stn, target)[0]


optimizer = make_optimizer(net)
scheduler = WarmupMultiStepLR(optimizer, (18, 30), 0.1, 1.0 / 3, 500, "linear")
net.train()

step = 0
best_map = -1
best_map_epoch = 0
best_rank1 = -1
best_rank1_epoch = 0
print("开始训练>>>")
for epoch in range(40):
    if epoch==0:
        print("开始测试直接迁移的结果")
        query_feature = FO.extract_cnn_feature(net, loader=query_loader, vis=False)
        gallery_feature = FO.extract_cnn_feature(net, loader=gallery_loader, vis=False)
        query_id, query_camera = query_loader.dataset.original_id, query_loader.dataset.cameras
        gallery_id, gallery_camera = gallery_loader.dataset.original_id, gallery_loader.dataset.cameras
        map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                            np.array(gallery_id), np.array(gallery_camera), vis=False)
        print("直接迁移的结果: map:{},rank-1:{},rank-5:{},rank-10:{}".format(map, cmc[0], cmc[4], cmc[9]))

    scheduler.step()
    for images, ids, cams in train_loader:
        feat, predict, feat_stn, predict_stn = net(images.cuda())
        loss_value = loss(feat, predict, feat_stn, predict_stn, ids.cuda())
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if step % 10 == 0:
            print(step, loss_value.item())
        step += 1
    # if (epoch + 1) > 12 and (epoch + 1) % 2 == 0:
    if (epoch + 1) % 2 == 0:
        save_network(save_path, net, epoch)
        print("第{}轮效果评估开始>>>".format(epoch + 1))
        query_feature = FO.extract_cnn_feature(net, loader=query_loader, vis=False)
        gallery_feature = FO.extract_cnn_feature(net, loader=gallery_loader, vis=False)
        query_id, query_camera = query_loader.dataset.original_id, query_loader.dataset.cameras
        gallery_id, gallery_camera = gallery_loader.dataset.original_id, gallery_loader.dataset.cameras
        map, cmc = market_evaluate.evaluate(query_feature, np.array(query_id), np.array(query_camera), gallery_feature,
                                            np.array(gallery_id), np.array(gallery_camera), vis=False)
        print("第{}轮训练结果: map:{},rank-1:{},rank-5:{},rank-10:{}".format(epoch + 1, map, cmc[0], cmc[4], cmc[9]))
        if map > best_map:
            best_map = map
            best_map_epoch = epoch
        if cmc[0] > best_rank1:
            best_rank1 = cmc[0]
            best_cmc_epoch = epoch

    print("已经训练了{}个epoch".format(epoch + 1))
print("最佳map:{},最佳rank-1{},最佳map训练轮数:{},最佳cmc训练轮数:{}".format(best_map, best_rank1, best_map_epoch, best_rank1_epoch))
