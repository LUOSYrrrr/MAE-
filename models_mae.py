# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        #image_size：图片的大小；
        #patch_size:把图片划分成小的patch，小的patch的尺寸；
        #num_classes:这次分类任务的类别总数；
        #channels:输入图片的通道数。

        # --------------------------------------------------------------------------
#1.encoder 模块 MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        print(self.patch_embed)
       #patch_size 应该是一个图片分出来的 一张有多大  inchans 一般都是3 图片层数嘛
       # embed——dim 这个是编出来的特征维度 1024
        num_patches = self.patch_embed.num_patches
        ###num_pathches 大小是x*y 就是图片分成x*y份
        #num_patches = (224/patch_size)**2 = 14 **2 = 196
       

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
         #是ViT并没有采用类似的pooling策略，
         # 而是直接增加一个特殊的class token，
         # 其最后输出的特征加一个linear classifier就可以实现对图像的分类
         # 所以输入ViT的sequence长度是class token对应的embedding在训练时随机初始化，然后通过训练得到，
         #1*1*2014
         # 加上class_token，零矩阵初始化，尺寸1*1*embed_dim.
         #第一个1是batchsize维度，是为了后面进行拼接所以设置成1。
         #第二、三个维度就是1*1024
         
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  
        self.blocks = nn.ModuleList([
            Block(embed_dim,num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        #这里的 block 就是VIT里的那个block  这个block也等到VIT代码时再讲
        #这里有几个他们用的小trick
        #nn.LayerNorm   #这个表示在channel 上做归一化 
        #nn.batchNorm  #这个是在batch上归一化
        #DropPath  # 这个也是一种与dropout不同的 drop方法
        #nn.GELU   #一种激活函数
        # --------------------------------------------------------------------------
#2.decoder模块 MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # 一个fc层 1024到512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        #一个mask编码 （1，1，512）

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        #一个位置编码 而且不训练 （1，197，512）
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim,decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
#解码器的注意力层只有8层 但也是12头的  输入是512维 
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        #预测层  512 到   256*3 （这个也不到224*224*3啊
        # --------------------------------------------------------------------------
#3.初始化模块
#3.2.1.3.1 找位置编码 
        self.norm_pix_loss = norm_pix_loss
        #norm_pix_loss=false

        self.initialize_weights()
        #第一个的值是false 等会看看有啥用  第二个是一个函数 我们进去看看 。

    def initialize_weights(self):
        print(self.patch_embed)
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        #1024，14，false-》得到（197，1024）
        #get_2d_sincos_pos_embed在util的pos_embed里
#3.2.1.3.2回到初始化
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
               
        #将numpy变为tensor 后 转float32 再扩充维度为（1，197，1024） 就得到了编码器的位置编码

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        ##512，14，false-》得到（197，1024）
        #解码器的位置编码  （1，197，512） 还是比编码器少了一半

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        #这个w是取出weight层的权重值。 正好可以看出 w的大小是 （1024，3，16，16） 1024是输出维度 3是输入维度 。
        # 相当于一个卷积 ？ 然后参数进行一个初始化 统一于 （1024， 3*16*16）正太分布 
        #mask 和 cls 也要初始化 。

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        #初始化其他层 self.apply应该是对遍历模型 对每一个模块 使用后面这个函数 我们进入初始化权重函数看一看 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        #首先进入这个函数  
        # p是一个小图的大小 
        # hw分别是yx方向图的个数  都是14 
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
       
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        #x 是（1，3，14，16，14，16） -（1，14，14，16，16，3）
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        #       然后reshape （1，14，14，16，16，3） -》(1,196,768) 此中过程 不足为外人道也 鬼知道你咋变的啊 。
        h = w = imgs.shape[2] // p
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        #p 16 h w, 14,14   
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        #x (1,196,768) -> (1,14,14,16,16,3) ->(1,3,14,16,14,16)  ->imgs(1,3,224,224) 
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        通过逐样变换执行逐样随机屏蔽。
        逐样本变换由argsort随机噪声完成。
        """
        '''
        首先 noise是随机生成的  比如说是 noise = [2,0,3,1] 
        然后 排序argsort: shuffle = [1,3,0,2]    到这里 是为了生成随机数  我们取前两个 也就是随机出来的1，3 
        作为mask的下标 
        对shuffle排序       ：  restore = [2，0，3，1]
        mask = [0,0,1,1]  我们根据restore对mask取数  得到[ 1,0,1,0]  下标1，3处就是0.            
        其实你可以把mask和shuffle看成一样的 你用restore对shuffle 取数 得到【0，1，2，3】发现是排序好的 。
         对【1，0，1，0】取数 得到[0,0,1,1]两个是对应起来的。
         '''

        print(x.shape)
        N, L, D = x.shape  # batch 1, length 196 , dim 1024
        len_keep = int(L * (1 - mask_ratio))#计算需要剩余多少片
        print('x.device=',x.device)
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        print('noise=',noise)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # 是对noise的值进行排序  ids_shuffle得到的是下标值。
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        #对排序后得的下标 再排序？这一步#我非常的不懂  后面看
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #保持噪声值小的那一堆？
        print('ids_keep shape',ids_keep.shape)
        print('ids_keep unsqueeze(-1) shape',ids_keep.unsqueeze(-1).shape)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
       #这个gather 就是在 x的 dim维 挑index的数。  但是好奇的是 这一串下来 不就是随机挑吗？
       # index的维度是 （1，49，1024）X是（1，196，1024） x_masked 是（1，49，1024）
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        #mask 是（1，196） 其中前49都是0 后面都是1

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        #到这里终于明白了  
        # 这个ids_REStore的作用 就是把mask当成noise 
        # 然后把mask按照#restore的位置排序  
        # 这样得到的mask就是一个  有mask的地方为1 没mask的地方为0的二维张量。
        return x_masked, mask, ids_restore
#4.2编码步骤 
    def forward_encoder(self, x, mask_ratio):
        #这里的mask这里非常难以理解 所以我举个例子 来看看 。 
        # embed patches
        print(x.shape)
        x = self.patch_embed(x)
        # #x:(1,3,224,224)->(1,196,1024)   14*14个片编码
        print(x.shape)

        # add pos embed w/o cls token
        print(self.pos_embed[:, 1:, :].shape)
        ## pos是1，197，1024 这里不要0的cls位置 
        # 位置信息是直接加到片编码上的  
        # 和我的想法很不一样  这样加上来真的会有效果么 。
        x = x + self.pos_embed[:, 1:, :]
        print(x.shape)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token 处理cls
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        ##cls加上位置信息 
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # # 这一句是为了防止批量的 也就是扩充复制 如果x的batch为N  cls也要复制N份
        x = torch.cat((cls_tokens, x), dim=1)
        # #x:(1,50,1024) ->(1,50,1024)   原来是扩充在片数这一维。

        # apply Transformer blocks
        #这里x要经历24个多头自注意力的磨练  然后归一化。
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
#4.3解码步骤 
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        #x  (1,50,1024) ->(1,50,512)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        ##ids_restore.shape[1] + 1 - x.shape[1] =196+1-50 =147也就是cls加片数减x=需要遮盖数
        #self.maskroken.shape = (1,1,512)  mask_tokens = (1,147,512) repeate是几就复制几

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)   # no cls token cls辛辛苦苦一辈子 
         #就这样没了  我还没看到你作用呢 麻烦半天  这里就是完成了 x和mask拼接后的X_
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        ## unshuffle    排序回去 按照 mask  index.shape = (1,196,512)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
        #得到了模型预测的图像结果  

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        #进入一个函数，这句就是把原来的图片 也编辑成（1，196，768）大小的 
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            #这个归一化 没进去 可能因为本来已经归过了 

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        #loss是像素差平方  然后对最后一维求平均 变成了 （1，196） 也就是每一个小pat 一个loss

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        #mask在相应没有遮盖的地方是0 所以就是只有遮盖的地方才求loss  返回loss值。回到run
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        #进froward第一句 就是这一句  我们接下来进入前向编码器里看一看 。
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        #回归forward  来到第二局 解码
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
