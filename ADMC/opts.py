import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Action Recognition')
    parser.add_argument('--dataset', default='MI', type=str, help='训练数据名称')#MIntRec
    parser.add_argument('--N', default=50, type=int, help='')
    parser.add_argument('--nEpochs', default=20, type=int, help='number of total epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default:32)')
    parser.add_argument('--lr', default=1e-5, type=float, help='initial learning rate (default:5e-4')
    # use_FeatureExtraction, use_DiffusionMlp
    parser.add_argument('--use_FeatureExtraction', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_Fusion', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_Fusion_GD', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_DDIM', action='store_true', help='当提供参数时，其值为 True,否则为Flase')

    parser.add_argument('--use_text', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_vid', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_wav', action='store_true', help='当提供参数时，其值为 True,否则为Flase')

    parser.add_argument('--use_modality', default=0, type=int, help='0,1,2,3,4,5')
    # 0 text missing
    # 1 vid missing
    # 2 wav missing
    # 3 text and vid missing
    # 4 text and wav missing
    # 5 vid and wav missing

    parser.add_argument('--use_DiffusionMlp', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_zero', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    
    parser.add_argument('--use_MMIR', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_MEIR', action='store_true', help='当提供参数时，其值为 True,否则为Flase')
    parser.add_argument('--use_full', action='store_true', help='当提供参数时，其值为 True,否则为Flase')

    args = parser.parse_args()

    # 检查逻辑1: --use_FeatureExtraction 为 True 时，其他指定参数中至少有一个为 True
    if args.use_FeatureExtraction:
        if args.use_Fusion:
            if (args.use_text and args.use_vid and args.use_wav):
                parser.error(
                    "--use_Fusion 为 True 时，--use_text, --use_vid 或 --use_wav 不能为 True")
        else:
            if not (args.use_text or args.use_vid or args.use_wav):
                parser.error(
                    "--当use_Fusion为False, 必须至少有一个参数--use_text, --use_vid 或 --use_wav 为 True")
    # # 检查逻辑2:
    else:
        if args.use_DiffusionMlp is True:
            pass
        else:
            task_mode = sum([args.use_MMIR, args.use_MEIR, args.use_zero])
            if task_mode > 1 or task_mode == 0:
                parser.error("Error: Only one of args.use_UniAug, args.use_MMIR, args.use_zero,必须选择一个模式")
       

    # 检查逻辑3: 只能有一个任务模式参数为 True
    # task_mode = sum([args.use_ADD, args.use_DiffusionMlp])
    # if task_mode > 1:
    #     parser.error("Error: Only one of --use_ADD, --use_DiffusionMlp")

    return args