import os
import yaml
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib.gridspec import GridSpec
from dataloader import Averager
from dataloader import weibo_data
from dataloader import clipdata2gpu
from sklearn.metrics import classification_report
from model.final_final_model import MultiModalFENDModel

import warnings
warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

class Trainer():
    def __init__(self,args,train_loader,test_loader,val_loader):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.loss_weight = [0.7, 0.1, 0.1, 0.1]
        self.label_smoothing = 0.1

    # def visualize_feature_space(self, features, labels, epoch, method='tsne'):
    #     """Enhanced feature space visualization with modern styling"""
    #     plt.figure(figsize=(10, 8))
    #
    #     # 使用更专业的颜色方案 (Tableau10)
    #     colors = plt.cm.tab10.colors
    #
    #     # 降维处理
    #     if method == 'pca':
    #         from sklearn.decomposition import PCA
    #         reducer = PCA(n_components=2)
    #     else:
    #         reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    #
    #     features_2d = reducer.fit_transform(features)
    #
    #     # 绘制参数优化
    #     scatter_params = {
    #         's': 15,  # 更小的点大小
    #         'alpha': 0.7,  # 适中的透明度
    #         'edgecolors': 'w',  # 白色边缘
    #         'linewidths': 0.3  # 细边缘线
    #     }
    #
    #     # 按类别绘制
    #     for class_id in np.unique(labels):
    #         plt.scatter(
    #             features_2d[labels == class_id, 0],
    #             features_2d[labels == class_id, 1],
    #             color=colors[class_id % len(colors)],
    #             label=f'Class {class_id}',
    #             **scatter_params
    #         )
    #
    #     # 样式增强
    #     plt.title(f'Feature Space Projection | Epoch {epoch} ({method.upper()})',
    #               fontsize=12, pad=20)
    #     plt.legend(fontsize=9, framealpha=0.9)
    #     plt.grid(True, alpha=0.2)
    #
    #     # 移除顶部和右侧边框
    #     for spine in plt.gca().spines.values():
    #         spine.set_visible(False)
    #
    #     # 保存高清图像
    #     plt.savefig(
    #         f'test_run/feature_space_epoch_{epoch}.png',
    #         dpi=300,
    #         bbox_inches='tight',
    #         transparent=False
    #     )
    #     plt.close()
    #
    # # 使用测试结果计算真实的混淆矩阵
    # # def plot_real_confusion_matrix(self, all_labels, all_preds, save_path=None):
    # #     """硬核兼容版混淆矩阵绘图（内部处理所有格式问题）"""
    # #     try:
    # #         # 暴力展平所有可能的嵌套结构
    # #         labels = np.concatenate([x.ravel() if isinstance(x, np.ndarray) else [x] for x in all_labels]).astype(int)
    # #         preds = np.concatenate([x.ravel() if isinstance(x, np.ndarray) else [x] for x in all_preds]).astype(int)
    # #
    # #         # 验证数据
    # #         assert len(labels) == len(preds), f"数据长度不匹配: labels={len(labels)}, preds={len(preds)}"
    # #         unique_values = set(np.unique(labels)) | set(np.unique(preds))
    # #         assert unique_values <= {0, 1, 2, 3}, f"非法类别值: {unique_values - {0, 1, 2, 3} }"
    # #
    # #         # 计算混淆矩阵
    # #         cm = confusion_matrix(labels, preds, normalize='true')
    # #
    # #         # 绘图配置
    # #         classes = ['H+T', 'L+T', 'H+F', 'L+F']
    # #         plt.figure(figsize=(10, 8), dpi=100)
    # #         plt.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    # #         plt.title('Normalized Confusion Matrix', pad=20, fontsize=16)
    # #         plt.colorbar(fraction=0.046, pad=0.04)
    # #
    # #         # 标注数值
    # #         thresh = 0.5
    # #         for i in range(cm.shape[0]):
    # #             for j in range(cm.shape[1]):
    # #                 plt.text(j, i, f"{cm[i, j]:.3f}",
    # #                          horizontalalignment="center",
    # #                          verticalalignment="center",
    # #                          color="white" if cm[i, j] > thresh else "black",
    # #                          fontsize=12)
    # #
    # #         # 坐标轴设置
    # #         plt.xticks(np.arange(4), classes, rotation=45, ha='right')
    # #         plt.yticks(np.arange(4), classes)
    # #         plt.xlabel('Predicted Label', labelpad=10)
    # #         plt.ylabel('True Label', labelpad=10)
    # #         plt.tight_layout()
    # #
    # #         # 保存结果
    # #         if save_path:
    # #             plt.savefig(save_path, bbox_inches='tight', dpi=300)
    # #             print(f"混淆矩阵已保存至: {save_path}")
    # #         plt.show()
    # #
    # #     except Exception as e:
    # #         print(f"绘图失败，请检查数据格式: {str(e)}")
    # #         print(f"labels样本数据: {all_labels[:3]}")
    # #         print(f"preds样本数据: {all_preds[:3]}")
    # def plot_real_confusion_matrix(self, all_labels, all_preds, save_path=None):
    #     """兼容所有Matplotlib版本的期刊级绘图"""
    #     try:
    #         # 1. 数据预处理（兼容您的append方式）
    #         labels = np.concatenate([x.ravel() for x in all_labels]).astype(int)
    #         preds = np.concatenate([x.ravel() for x in all_preds]).astype(int)
    #
    #         # 2. 创建图像（不依赖任何style）
    #         plt.figure(figsize=(5, 4))
    #         plt.rcParams.update({
    #             'font.family': 'serif',
    #             'axes.grid': False,
    #             'axes.edgecolor': 'black',
    #             'axes.linewidth': 0.8
    #         })
    #
    #         # 3. 计算混淆矩阵
    #         cm = confusion_matrix(labels, preds, normalize='true')
    #         classes = ['H+T', 'L+T', 'H+F', 'L+F']  # 请确认类别顺序
    #
    #         # 4. 绘制热力图
    #         im = plt.imshow(cm, cmap='Blues', vmin=0, vmax=1)
    #
    #         # 5. 添加色标
    #         cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    #         cbar.set_label('Accuracy', rotation=270, labelpad=15)
    #
    #         # 6. 标注数值
    #         for i in range(cm.shape[0]):
    #             for j in range(cm.shape[1]):
    #                 plt.text(j, i, f"{cm[i, j]:.3f}",
    #                          ha="center", va="center",
    #                          color="white" if cm[i, j] > 0.5 else "black",
    #                          fontsize=9)
    #
    #         # 7. 坐标轴设置
    #         plt.xticks(np.arange(4), classes, rotation=45, ha='right')
    #         plt.yticks(np.arange(4), classes)
    #         plt.xlabel("Predicted Label", labelpad=10)
    #         plt.ylabel("True Label", labelpad=10)
    #         plt.tight_layout()
    #
    #         # 8. 保存图像（自动处理路径）
    #         if save_path:
    #             base_path = os.path.splitext(save_path)[0]
    #             plt.savefig(f"{base_path}.pdf", bbox_inches='tight')
    #             plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
    #             print(f"图像已保存至: {base_path}.pdf/png")
    #         plt.show()
    #
    #     except Exception as e:
    #         print(f"绘图错误: {str(e)}")
    #         print("调试信息:")
    #         print(f"标签样例: {labels[:5]}")
    #         print(f"预测样例: {preds[:5]}")
    def train(self):
        self.model = MultiModalFENDModel(
            self.args.model_name,
            self.args.emb_dim,
            self.args.mlp_dims,
            self.args.bert,
            self.args.dropout
        )
        self.model.cuda()

        os.makedirs(self.args.save_dir, exist_ok=True)
        # 寻找可用的版本号目录
        version = 0
        while True:
            version_dir = os.path.join(self.args.save_dir, f'{self.args.name}_v{version}')
            if not os.path.exists(version_dir):
                self.args.save_dir = version_dir
                os.makedirs(version_dir)  # 创建版本目录
                break
            version += 1
        # 在版本目录下创建子目录和配置文件
        os.makedirs(os.path.join(self.args.save_dir, 'runs'), exist_ok=True)
        with open(os.path.join(self.args.save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f, sort_keys=False)
        #训练配置
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                lr=self.args.lr,
                                betas=(self.args.beta1, self.args.beta2),
                                eps=self.args.eps,
                                weight_decay=self.args.weight_decay)
        num_batches_per_epoch = len(self.train_loader)
        warmup_steps = self.args.warmup_steps
        total_steps = self.args.total_epoch * num_batches_per_epoch - warmup_steps

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,total_steps,eta_min=self.args.lr/10)

        max_avg_acc = 0
        class_weight = torch.tensor([1.0,1.0,0.8,0.8]).cuda()
        loss_fn = nn.CrossEntropyLoss(weight = class_weight)
        # loss_fn = torch.nn.BCELoss()
        for epoch in range(self.args.total_epoch):
            self.model.train()
            avg_loss = Averager()
            train_data_iter = tqdm.tqdm(enumerate(self.train_loader),total = len(self.train_loader))
            print(('\n' + '%11s' * 5) % ('Epoch', 'GPU_mem', 'Cur_loss', 'avg_loss', 'lr'))
            for i,batch in train_data_iter:
                #学习率预热
                current_step = epoch * num_batches_per_epoch + i
                if current_step < warmup_steps:
                    lr = self.args.lr * current_step / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr


                batch = clipdata2gpu(batch)
                encoded_batch = {
                    "input_ids": batch['input_ids'],
                    "attention_mask": batch['attention_mask'],
                }
                labels = batch['label'].long()
                optimizer.zero_grad()

                final_pred, fusion_pred, image_pred, text_pred = self.model(encoded_batch,**batch)
                loss0 = loss_fn(final_pred, labels)
                loss1 = loss_fn(fusion_pred, labels)
                loss2 = loss_fn(image_pred, labels)
                loss3 = loss_fn(text_pred, labels)

                loss = sum(w * l for w, l in zip(self.loss_weight, [loss0, loss1, loss2, loss3]))
                loss.backward()
                optimizer.step()
                # 更新学习率
                if current_step >= warmup_steps:
                    scheduler.step()

                avg_loss.add(loss.item())

               # 显示进度
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else '0G'
                train_data_iter.set_description(
                    ('%11s' * 2 + '%11.4g' * 3) %
                    ( f'{epoch + 1}/{self.args.total_epoch}',mem,loss.item(), avg_loss.item(), optimizer.param_groups[0]['lr'])#
                )

            #测试
            self.model.eval()
            all_labels = []
            all_preds = []
            test_loss = 0
            right_num , tot_num = 0,0
            with torch.no_grad():
                test_data_iter = tqdm.tqdm(enumerate(self.test_loader),total = len(self.test_loader))
                print(('\n' + '%11s' * 6) % ('Epoch', 'GPU_mem', 'Cur_acc', 'avg_acc', 'avg_loss','cur_right_num'))
                for i,batch in test_data_iter:
                    batch = clipdata2gpu(batch)
                    encoded_batch = {
                        "input_ids": batch['input_ids'],
                        "attention_mask": batch['attention_mask'],
                    }
                    labels = batch['label'].long()

                    final_pred, fusion_pred, image_pred, text_pred = self.model(encoded_batch,**batch)

                    temp = (torch.argmax(final_pred,dim=1) == labels).sum().item()
                    # print(torch.argmax(final_pred,dim=1))
                    cur_right_num = temp
                    cur_num = labels.shape[0]
                    right_num += cur_right_num
                    tot_num += cur_num
                    test_loss0 = loss_fn(final_pred, labels)
                    test_loss1 = loss_fn(fusion_pred, labels)
                    test_loss2 = loss_fn(image_pred, labels)
                    test_loss3 = loss_fn(text_pred, labels)
                    cur_test_loss = sum(w * l for w, l in zip(self.loss_weight, [test_loss0, test_loss1, test_loss2, test_loss3]))
                    test_loss = (test_loss * i + cur_test_loss.item()) / (i + 1)
                    mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else '0G'
                    test_data_iter.set_description(
                        ('%11s' * 2 + '%11.4g' * 4) % 
                        (f'{epoch + 1}/{self.args.total_epoch}', mem, cur_right_num / cur_num, right_num / tot_num, test_loss, temp )
                    )
                    all_labels.append(labels.cpu().numpy())
                    all_preds.append(final_pred.argmax(dim = 1).cpu().numpy())

            # 合并所有batch的结果
            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)
            # 生成分类报告
            target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']  # 替换为你的类别名称
            report = classification_report(
                all_labels, 
                all_preds, 
                target_names=target_names,
                digits=4  # 控制小数点位数
            )
            print("Classification Report:\n", report)
            avg_acc = right_num / tot_num
            if avg_acc > max_avg_acc:
                max_avg_acc = avg_acc
                torch.save(self.model.state_dict(),os.path.join(self.args.save_dir,'model_best.pth'))
                print(f'Save model to {os.path.join(self.args.save_dir,"model_best.pth")}')
        #最终测试
        self.model.load_state_dict(torch.load(os.path.join(self.args.save_dir,'model_best.pth')))
        self.model.eval()
        right_num, tot_num = 0, 0
        with torch.no_grad():
            val_data_iter = tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            print(('\n' + '%11s' * 4) % ('Epoch', 'GPU_mem', 'Cur_acc', 'avg_acc'))
            for i, batch in val_data_iter:
                batch = clipdata2gpu(batch)
                encoded_batch = {
                    "input_ids": batch['input_ids'],
                    "attention_mask": batch['attention_mask'],
                }
                labels = batch['label'].long()
                final_pred, fusion_pred, image_pred, text_pred = self.model(encoded_batch, **batch)
                cur_right_num = (torch.argmax(final_pred, dim=1) == labels).sum().item()
                cur_num = labels.shape[0]
                right_num += cur_right_num
                tot_num += cur_num
                mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G' if torch.cuda.is_available() else '0G'
                test_data_iter.set_description(
                    ('%11s' * 2 + '%11.4g' * 2) %
                    (f'final_val', mem, cur_right_num / cur_num, right_num / tot_num)
                )



class Run():
    def __init__(self,args):
        self.args = args
    def get_loader(self):
        self.train_path = self.args.path+'train_origin.jsonl'
        self.test_path = self.args.path+'test_origin.jsonl'
        self.val_path = self.args.path+'dev_origin.jsonl'
        loader = weibo_data(max_len=self.args.max_len, batch_size=self.args.batch_size, model_name = self.args.model_name ,num_workers=self.args.num_workers,vocab_file=self.args.vocab_file)
        train_data = loader.load_data(self.train_path,r"D:\my\test_MMDFND\数据集\train_loader.pkl",r"D:\my\test_MMDFND\数据集\new_train_clip_crops_loader.pkl",shuffle=True)
        test_data = loader.load_data(self.test_path,r"D:\my\test_MMDFND\数据集\test_loader.pkl",r"D:\my\test_MMDFND\数据集\new_test_clip_crops_loader.pkl",shuffle=False)
        val_data = loader.load_data(self.val_path,r"D:\my\test_MMDFND\数据集\val_loader.pkl",r"D:\my\test_MMDFND\数据集\new_val_clip_crops_loader.pkl",shuffle=False)

        return train_data,test_data,val_data
    def main(self):
        train_loader,test_loader,val_loader = self.get_loader()
        trainer = Trainer(self.args,train_loader,test_loader,val_loader)

        trainer.train()

