import torch
from opts import parse_opts
args = parse_opts()


if args.dataset == 'MIntRec':#
    def load_checkpoint_FE(s_dict,layer_name):
        # 加载t
        # checkpoint = torch.load("./bertMy_t_7258.pth")
        checkpoint = torch.load("./best_MI_text_7236.pth")
        print('加载MIntRec t模态特征提取网络')
        for name in checkpoint:
            name_ = "feature_extraction_model.netL." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)

        # 加载v
        print('加载MIntRec v模态特征提取网络')

        # checkpoint = torch.load("./bertMy_v_1303.pth")
        # for name in s_dict:
        #     if "netV" in name:
        #         name_ = name.replace("feature_extraction_model.netV","Vid_model")
        #         if name_ in checkpoint:
        #             print('name_1',name_)
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)
        #     if 'fc' in name:
        #         name_ = name.replace("MF_model.", "")
        #         if name_ in checkpoint:
        #             print('name_2', name_)
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)

        # checkpoint = torch.load("./bert_MI_vid_1506.pth")
        checkpoint = torch.load("./best_MI_vid_1461.pth")###与wav融合更好
        for name in checkpoint:
            name_ = "feature_extraction_model.netV." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)


        # 加载a
        print('加载MIntRec a模态特征提取网络')

        # checkpoint = torch.load("./bertMy_a_3281.pth")
        # for name in s_dict:
        #     if "netA" in name:
        #         name_ = name.replace("feature_extraction_model.netA","Wav_model")
        #         if name_ in checkpoint:
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)
        #     if 'fc' in name:
        #         name_ = name.replace("MF_model.", "")
        #         if name_ in checkpoint:
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)

        # checkpoint = torch.load("./bert_MI_wav_TF_3281.pth")
        checkpoint = torch.load("./best_MI_wav_3348.pth")
        for name in checkpoint:
            name_ = "feature_extraction_model.netA." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)


        return s_dict,layer_name


    def load_checkpoint_fusion_FE(s_dict, layer_name):
        # 加载feature_extraction_model
        print('加载MIntRec feature_extraction_model')
        checkpoint = torch.load("./best_MI_fusion_7124.pth")#
        for name in checkpoint:
            name_ = "feature_extraction_model." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)
        return s_dict, layer_name

else:
    def load_checkpoint_FE(s_dict,layer_name):
        # checkpoint = torch.load("./bertMy_IE_t_700.pth")#IEMOCAP
        # for name in s_dict:
        #     if "netL" in name:
        #         name_ = name.replace("feature_extraction_model.","")
        #         if name_ in checkpoint:
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)

        checkpoint = torch.load("./best_IE_text_7000.pth")  # IEMOCAP bert_IE_text_7000.pth
        print('加载IEMOCAP t模态特征提取网络')
        for name in checkpoint:
            name_ = "feature_extraction_model.netL." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)

        # 加载v
        # checkpoint = torch.load("./bertMy_IE_v_600.pth")#IEMOCAP
        # for name in s_dict:
        #     if "netV" in name:
        #         name_ = name.replace("feature_extraction_model.","")
        #         if name_ in checkpoint:
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)

        checkpoint = torch.load("./best_IE_vid_5817.pth")#IEMOCAP bert_IE_vid_5817.pth
        print('加载IEMOCAP v模态特征提取网络')
        for name in checkpoint:
            name_ = "feature_extraction_model.netV." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)

        # 加载a
        # checkpoint = torch.load("./bertMy_IE_a_581.pth")#IEMOCAP
        # for name in s_dict:
        #     if "netA" in name:
        #         name_ = name.replace("feature_extraction_model.","")
        #         if name_ in checkpoint:
        #             s_dict[name] = checkpoint[name_]
        #             layer_name.append(name)

        checkpoint = torch.load("./best_IE_wav_5800.pth")  # IEMOCAP  bert_IE_wav_5800.pth
        print('加载IEMOCAP a模态特征提取网络')
        for name in checkpoint:
            name_ = "feature_extraction_model.netA." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)

        return s_dict,layer_name

    def load_checkpoint_fusion_FE(s_dict, layer_name):
        # 加载feature_extraction_model
        print('加载IEMOCAP feature_extraction_model')
        checkpoint = torch.load("./best_IE_fusion_7917.pth")#7558
        for name in checkpoint:
            name_ = "feature_extraction_model." + name
            if name_ in s_dict:
                s_dict[name_] = checkpoint[name]
                layer_name.append(name_)
        return s_dict,layer_name


# 加载并继续训练MLP
def load_checkpoint_MLP(s_dict,layer_name):
    if args.dataset == 'IEMOCAP':
        # checkpoint = torch.load("./bert_IE_mlp_full_0360.pth")###full
        # checkpoint = torch.load("./bert_IE_mlp_3f_718.pth")
        # checkpoint = torch.load("./bert_IE_mlp_TF_0256.pth")
        # checkpoint = torch.load("./bert_IE_mlp_TF_full_0342.pth")
        #3倍
        # checkpoint = torch.load("./bert_IE_mlp_TF_0220.pth")
        if args.use_Fusion:
            checkpoint = torch.load("./best_IE_Fusion_mlp_0092.pth")#0099
        else:
            # checkpoint = torch.load("./best_IE_mlp_0320_unet.pth")#   TF:best_IE_mlp_0212.pth  best_IE_mlp_0.320_unet.pth
            # checkpoint = torch.load("./best_IE_mlp_0330_unetMF.pth")
            checkpoint = torch.load("./best_IE_mlp.pth")
            print('******')
            # checkpoint = torch.load("./best_IE_mlp_0280_MF.pth")  #
            # checkpoint = torch.load("./best_IE_mlp_0300_TF.pth")  #
    else:
        if args.use_Fusion:
            checkpoint = torch.load("./best_MI_Fusion_mlp_0100.pth")
        else:
            # checkpoint = torch.load("./best_MI_mlp_0150_unet.pth")#  U-net:best_MI_mlp_0140.pth   TF:best_MI_mlp_0110.pth
            checkpoint = torch.load("./best_MI_mlp.pth")  # U-net:best_MI_mlp_0140.pth   TF:best_MI_mlp_0110.pth

    for name in checkpoint:
        name_ = "CDMC_model." + name
        if name_ in s_dict:
            s_dict[name_] = checkpoint[name]
            if not args.use_DiffusionMlp:
                layer_name.append(name_)###############如果只加载
    return s_dict, layer_name


# 加载并固定网络参数
def load_GD(model,s_dict,layer_name):
    # 加载
    model.load_state_dict(s_dict)
    if args.use_FeatureExtraction and not args.use_Fusion:
        pass
    else:
        # 固定
        for (name, param) in model.named_parameters():
            if name in layer_name:  #
                # print(name, "被固定的网络层")
                param.requires_grad = False  # 固定
            else:
                print(name, "被放开的网络层")
    return model


