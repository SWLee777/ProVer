import models_mae
from .pivot import *
from .layers import *
from cn_clip.clip import load_from_name
from transformers import BertModel, AutoTokenizer, AutoModel

class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name, output_hidden_states=False):
        super(TextEmbeddingModel, self).__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)#√允许自定义模型架构
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)#'princeton-nlp/unsup-simcse-roberta-base'基于 RoBERTa 的无监督 SimCSE（对比学习）模型

    def pooling(self, model_output, attention_mask):#带注意力掩码的平均池化（Masked Mean Pooling），用于从Transformer模型的序列输出中提取句子级别的表示。
        model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        return emb

    def forward(self, encoded_batch):
        model_output = self.model(**encoded_batch)
        model_output = model_output["last_hidden_state"]#√
        emb = self.pooling(model_output, encoded_batch['attention_mask'])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class CrossModule4Batch(nn.Module):#通过交叉注意力机制计算模态间相关性，并输出融合后的特征
    def __init__(self, text_in_dim=512, image_in_dim=512, corre_out_dim=320):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = text_in_dim
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2)
        image_in = image.unsqueeze(1)
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        return torch.sum(x, (1)) / (x.shape[1])

    def sigma(self, x):
        return torch.sqrt(
            (torch.sum((x.permute([1, 0]) - self.mu(x)).permute([1, 0]) ** 2, (1)) + 0.000000023) / (x.shape[1]))

    def forward(self, x, mu, sigma):
        x_mean = self.mu(x)
        x_std = self.sigma(x)
        x_reduce_mean = x.permute([1, 0]) - x_mean
        x_norm = x_reduce_mean / x_std
        return (sigma.squeeze(1) * (x_norm + mu.squeeze(1))).permute([1, 0])


class MultiModalFENDModel(torch.nn.Module):
    def __init__(self, model_name ,emb_dim, mlp_dims, bert, dropout):
        super(MultiModalFENDModel, self).__init__()

        self.num_expert = 6
        self.unified_dim = emb_dim#768
        self.text_dim,self.image_dim = 768, 768

        #文本模型..
        self.text_model = TextEmbeddingModel(model_name).requires_grad_(False)
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        #mae模型
        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cuda',weights_only=True)#
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False

        #clip模型
        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        self.ClipModel.requires_grad_(False)

        feature_kernel = {1:64,2:64,3:64,5:64,10:64}

        #注意力机制
        # self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        # self.final_attention = TokenAttention(320)#faison

        #专家网络（input_dim，320）
        #多尺度一维卷积特征提取器
        self.text_experts = nn.ModuleList([cnn_extractor(self.text_dim,feature_kernel) for _ in range(self.num_expert)])
        self.image_experts = nn.ModuleList([cnn_extractor(self.image_dim, feature_kernel) for _ in range(self.num_expert)])
        self.fusion_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(320, 320),
                nn.SiLU(),
                nn.Linear(320, 320)
            ) for _ in range(self.num_expert)
        ])
        self.final_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(320, 320),
                nn.GELU(),
                nn.Linear(320, 320)
            ) for _ in range(self.num_expert)
        ])

        #门控网络
        self.text_gate = nn.Sequential(
            nn.Linear(self.unified_dim ,self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim,self.num_expert),
            nn.Dropout(0.1),
            nn.Softmax(dim = 1)
        )
        self.image_gate = nn.Sequential(
            nn.Linear(self.unified_dim , self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.num_expert),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(320, 160),
            nn.SiLU(),
            nn.Linear(160, self.num_expert),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )
        self.final_gate = nn.Sequential(
            nn.Linear(320, 160),
            nn.SiLU(),
            nn.Linear(160, 80),
            nn.SiLU(),
            nn.Linear(80, self.num_expert),
            nn.Dropout(0.1),
            nn.Softmax(dim=1)
        )

        #分类器
        self.text_classifier = MLP(320, [128, 64], dropout)
        self.image_classifier = MLP(320, [128, 64], dropout)
        self.fusion_classifier = MLP(320, [128, 64], dropout)
        self.final_classifier = MLP(320, [128, 64], dropout)

        #特征融合
        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)#input_dim,out_dim,embed_dims,dropout
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)

        #自适应实例归一化
        self.mapping_T_MLP_mu = nn.Sequential(
            nn.Linear(4, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.mapping_T_MLP_sigma = nn.Sequential(
            nn.Linear(4, self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, 1),
        )
        self.adaIN = AdaIN()

        #povit
        self.feature_num = 4  # 特征序列长度
        self.layers = 12      # Transformer层数
        self.active = nn.SiLU()
        self.dropout2 = nn.Dropout(0.2)

        self.mlp_img = nn.ModuleList([MLP_trans(320, 320, dropout) for _ in range(self.feature_num)])
        self.mlp_text = nn.ModuleList([MLP_trans(320, 320, dropout) for _ in range(self.feature_num)])
        self.mlp_fusion = nn.ModuleList([MLP_trans(320, 320, dropout) for _ in range(self.feature_num)])
        self.transformers = nn.ModuleList([
            TransformerLayer(320, head_num=4,dropout=0.6,attention_dropout=0,initializer_range=0.02)
            for _ in range(self.layers)
        ])
        self.mlp_star_f1 = nn.Linear(320 * 4, 320)#拼接四个子特征后降维
        self.mlp_star_f2 = nn.Linear(320, 320)#进一步压缩提炼

        self.cross_feature_model = CrossModule4Batch()


    def fusion_img_text(self, image_emb, text_emb, fusion_emb, mlp_img, mlp_text, mlp_fusion, transformers, mlp_star_f1,mlp_star_f2):
        #四个MLP生成四个子特征（batch_size,4,320）
        for text_feature_num in range(0, self.feature_num):
            if text_feature_num == 0:
                text_feature_seq = mlp_text[text_feature_num](text_emb)
                text_feature_seq = text_feature_seq.unsqueeze(1)
            else:
                text_feature_seq = torch.cat((text_feature_seq, mlp_text[text_feature_num](text_emb).unsqueeze(1)), 1)

        for img_feature_num in range(0, self.feature_num):
            if img_feature_num == 0:
                img_feature_seq = mlp_img[img_feature_num](image_emb)
                img_feature_seq = img_feature_seq.unsqueeze(1)
            else:
                img_feature_seq = torch.cat((img_feature_seq, mlp_img[img_feature_num](image_emb).unsqueeze(1)), 1)

        for fusion_feature_num in range(0, self.feature_num):
            if fusion_feature_num == 0:
                fusion_feature_seq = mlp_fusion[fusion_feature_num](fusion_emb)
                fusion_feature_seq = fusion_feature_seq.unsqueeze(1)
            else:
                fusion_feature_seq = torch.cat((fusion_feature_seq, mlp_fusion[fusion_feature_num](fusion_emb).unsqueeze(1)), 1)

        # print(img_feature_seq.shape)
        # print(text_feature_seq.shape)
        # print(fusion_feature_seq.shape)

        # star_emb1 = (img_feature_seq[:, 0, :] + text_feature_seq[:, 0, :] + fusion_feature_seq[:, 0, :]) / 3
        # star_emb2 = (img_feature_seq[:, 1, :] + text_feature_seq[:, 1, :] + fusion_feature_seq[:, 1, :]) / 3
        # star_emb3 = (img_feature_seq[:, 2, :] + text_feature_seq[:, 2, :] + fusion_feature_seq[:, 2, :]) / 3
        # star_emb4 = (img_feature_seq[:, 3, :] + text_feature_seq[:, 3, :] + fusion_feature_seq[:, 3, :]) / 3
        #初始化四个子特征
        star_emb1 = text_feature_seq[:, 0, :]
        star_emb2 = text_feature_seq[:, 1, :]
        star_emb3 = text_feature_seq[:, 2, :]
        star_emb4 = text_feature_seq[:, 3, :]

        for sa_i in range(0, int(self.layers), 3):
            #text
            trans_text_item = torch.cat(
                [star_emb1.unsqueeze(1),
                 star_emb2.unsqueeze(1),
                 star_emb3.unsqueeze(1),
                 star_emb4.unsqueeze(1),
                 text_feature_seq], dim = 1)
            #trans_text_item : (batch_size,8,320)
            text_output = transformers[sa_i + 2](trans_text_item)#（batch_size，8，320）
            #更新star特征（残差连接+平均）
            star_emb1 = (text_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (text_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (text_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (text_output[:, 3, :] + star_emb4) / 2
            #更新文本的特征序列
            text_feature_seq = text_output[:, 4:self.feature_num + 4, :] + text_feature_seq#(batch_size,4,320)

            #image
            trans_img_item = torch.cat(
                [star_emb1.unsqueeze(1),
                 star_emb2.unsqueeze(1),
                 star_emb3.unsqueeze(1),
                 star_emb4.unsqueeze(1),
                 img_feature_seq], 1)
            img_output = transformers[sa_i + 1](trans_img_item)
            star_emb1 = (img_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (img_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (img_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (img_output[:, 3, :] + star_emb4) / 2
            img_feature_seq = img_output[:, 4:self.feature_num + 4, :] + img_feature_seq

            #fusion
            trans_fusion_item = torch.cat(
                [star_emb1.unsqueeze(1),
                 star_emb2.unsqueeze(1),
                 star_emb3.unsqueeze(1),
                 star_emb4.unsqueeze(1),
                 fusion_feature_seq], 1)
            fusion_output = transformers[sa_i](trans_fusion_item)
            star_emb1 = (fusion_output[:, 0, :] + star_emb1) / 2
            star_emb2 = (fusion_output[:, 1, :] + star_emb2) / 2
            star_emb3 = (fusion_output[:, 2, :] + star_emb3) / 2
            star_emb4 = (fusion_output[:, 3, :] + star_emb4) / 2
            fusion_feature_seq = fusion_output[:, 4:self.feature_num + 4, :] + fusion_feature_seq

        item_emb_trans = self.dropout2(torch.cat([star_emb1, star_emb2, star_emb3, star_emb4], 1))
        item_emb_trans = self.dropout2(self.active(mlp_star_f1(item_emb_trans)))
        item_emb_trans = self.dropout2(self.active(mlp_star_f2(item_emb_trans)))
        # item_emb = torch.cat([star_emb1, star_emb2, star_emb3, star_emb4], 1)
        # item_emb = self.mlp_star_f2(F.silu(self.mlp_star_f1(item_emb)))
        return item_emb_trans

    def forward(self,new_text,**kwargs):
        inputs = kwargs['token_ids']
        masks = kwargs['masks']
        image = kwargs['image']#(batch,3,224,224)

        text_atn_feature = self.text_model(new_text)#([batch,768])
        text_feature = self.bert(inputs,attention_mask=masks)[0]#([batch, 320, 768])

        image_feature = self.image_model.forward_ying(image)#(([batch, 197, 768])(mae)
        image_atn_feature, _ = self.image_attention(image_feature)#(batch,768)

        #clip特征提取
        clip_image = kwargs['clip_image']#(batch,3,224,224)
        clip_text = kwargs['clip_text']#(batch,197)

        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(clip_image)# ([batch, 512])
            clip_text_feature = self.ClipModel.encode_text(clip_text)  # ([batch, 512])
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)# ([batch, 512])
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)# ([batch, 512])
        clip_fusion_feature = self.cross_feature_model(clip_text_feature.float(),clip_image_feature.float())#(batch,320)

        # 专家处理
        text_gate_out = self.text_gate(text_atn_feature)#(batch,6)
        text_expert_gate_output = sum(
            expert(text_feature) * text_gate_out[:, i].unsqueeze(1)
            for i, expert in enumerate(self.text_experts)
        )#(batch,320)

        image_gate_out = self.image_gate(image_atn_feature)#(batch,6)
        image_expert_gate_output = sum(
            expert(image_feature) * image_gate_out[:, i].unsqueeze(1)
            for i, expert in enumerate(self.image_experts)
        )#(batch,320)

        #融合特征（简单融合）
        fusion_feature = torch.cat((clip_fusion_feature, text_expert_gate_output, image_expert_gate_output), dim=-1)#(batch,960)
        fusion_feature = self.MLP_fusion(fusion_feature)#(batch,320)

        fusion_gate_out = self.fusion_gate(fusion_feature)#(batch,6)
        fusion_expert_gate_output = sum(
            expert(fusion_feature) * fusion_gate_out[:, i].unsqueeze(1)
            for i, expert in enumerate(self.fusion_experts)
        )#(batch,320)

        #深度融合
        cross_knowledge = self.fusion_img_text(text_expert_gate_output, image_expert_gate_output,
                                               fusion_expert_gate_output, self.mlp_img, self.mlp_text,
                                               self.mlp_fusion, self.transformers, self.mlp_star_f1,
                                               self.mlp_star_f2)#(batch,320)

        final_score = torch.sigmoid(self.fusion_classifier(cross_knowledge))#(batch,4)
        mu = self.mapping_T_MLP_mu(final_score)#(batch,1)
        sigma = self.mapping_T_MLP_sigma(final_score)#(batch,1)
        final_temp_feature = self.adaIN(cross_knowledge,mu,sigma)#(batch,320)

        final_gate_out = self.final_gate(final_temp_feature)#(batch,6)
        final_expert_gate_output = sum(
            expert(final_temp_feature) * final_gate_out[:, i].unsqueeze(1)
            for i, expert in enumerate(self.final_experts)
        )#(batch,320)


        #分类
        text_pred = torch.sigmoid(self.text_classifier(text_expert_gate_output).squeeze(1))
        image_pred = torch.sigmoid(self.image_classifier(image_expert_gate_output).squeeze(1))
        fusion_pred = torch.sigmoid(self.fusion_classifier(fusion_expert_gate_output).squeeze(1))
        final_pred = torch.sigmoid(self.final_classifier(final_expert_gate_output).squeeze(1))
        return final_pred, fusion_pred, image_pred, text_pred




