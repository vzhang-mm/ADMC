import torch
from diffusers import DDPMScheduler,DDIMScheduler
from opts import parse_opts

args = parse_opts()
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 定义前向和反向过程
class DiffusionModel:
    def __init__(self, beta_start, beta_end, num_steps):
        self.num_steps = num_steps
        if args.use_DDIM:
            self.noise_scheduler = DDIMScheduler(beta_start=beta_start, beta_end=beta_end, num_train_timesteps=num_steps,beta_schedule='linear')  # squaredcos_cap_v2
            '''
            选择性地从1000步中选取50个关键时间步进行反向去噪。
            这50步代表了从噪声生成到原始数据的整个过程的一个子集，通过这些步数，模型能够快速且有效地生成高质量的图像。
            '''
            self.num_inference_steps=50
            self.noise_scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)
        else:
            self.noise_scheduler = DDPMScheduler(beta_start=beta_start, beta_end=beta_end, num_train_timesteps=num_steps, beta_schedule='linear')#squaredcos_cap_v2

    def q_sample(self, x_start, t, noise):  # 全域加噪
        x_noised = self.noise_scheduler.add_noise(x_start, noise, t)
        return x_noised


    def p_sample(self, model, x, t, mask):  # 去噪
        model.eval()
        with torch.no_grad():
            # 利用所有模态数据来预测噪声
            pred_noise = model(x, t)  # 预测 t 时刻的噪声
        updated_x = self.noise_scheduler.step(pred_noise, t, x).prev_sample

        if args.use_full:
            pass
        else:
            #保持已有模态特征（条件）不变
            mask = mask.unsqueeze(-1).expand(x.size())
            updated_x = torch.where(mask, updated_x, x)  # mask为True的位置用去噪后的x代替，原已有特征值保持不变

        return updated_x

    def generate_samples(self, model, x_init, num_steps= 500, mask=None):
        samples = x_init  # 使用原类别的数据初始化
        for t in reversed(range(num_steps)):
            # print(t)
            samples = self.p_sample(model, samples, t, mask)
        return samples

    # def p_sample(self, model, x, y, t):  # 去噪
    #     model.eval()
    #     with torch.no_grad():
    #         # 利用所有模态数据来预测噪声
    #         pred_noise = model(x, t, y)  # 预测 t 时刻的噪声
    #     x = self.noise_scheduler.step(pred_noise, t, x).prev_sample
    #     return x
    #
    # def generate_samples(self, model, x, y, num_steps = 500):
    #     samples = x  # 使用原类别的数据初始化
    #     for t in reversed(range(num_steps)):
    #         samples = self.p_sample(model, samples, y, t)
    #     return samples



# 定义前向和反向过程
# class DiffusionModel:
#     def __init__(self, beta_start, beta_end, num_steps):
#         '''
#         beta_start: 扩散过程的起始beta值。
#         beta_end: 扩散过程的结束beta值。
#         num_steps: 扩散过程的步骤数。
#         self.betas: 从 beta_start 到 beta_end 线性插值生成的噪声强度序列。
#         self.alpha_hat: 计算并保存 1 - betas 的累积乘积，用于生成和去噪过程中。
#         '''
#         self.num_steps = num_steps
#         self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
#         self.alphas = 1 - self.betas
#         self.alpha_hat = torch.cumprod(self.alphas, dim=0).to(device)  # 用于计算张量中元素的累积乘积
#
#
#     def q_sample(self, x_start, t, noise):  # 全域加噪
#         sqrt_alpha_hat_t = torch.sqrt(self.alpha_hat[t]).to(device)
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alpha_hat[t]).to(device)
#
#         if x_start.shape == 3:
#             sqrt_alpha_hat_t = sqrt_alpha_hat_t.unsqueeze(1).unsqueeze(1)
#             sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t.unsqueeze(1).unsqueeze(1)
#         else:
#             sqrt_alpha_hat_t = sqrt_alpha_hat_t.unsqueeze(1)
#             sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t.unsqueeze(1)
#
#         x_noised = sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise
#
#         return x_noised
#
#     def p_sample(self, model, x, t):  # 去噪
#         '''
#         model: 用于预测噪声的模型
#         x: 当前噪声数据
#         mask: 噪声掩码
#         t: 当前时间步
#         '''
#         model.eval()
#         with torch.no_grad():
#             # 利用所有模态数据来预测噪声
#             pred_noise = model(x.to(device), t)  # 预测 t 时刻的噪声
#
#         sqrt_alpha_t = torch.sqrt(self.alphas[t]).to(device)
#         sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - self.alpha_hat[t]).to(device)
#         beta_t = self.betas[t].to(device)
#
#         if x.shape == 3:
#             sqrt_alpha_t = sqrt_alpha_t.unsqueeze(1).unsqueeze(1)
#             sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t.unsqueeze(1).unsqueeze(1)
#             beta_t =  beta_t.unsqueeze(1).unsqueeze(1)
#         else:
#             sqrt_alpha_t = sqrt_alpha_t.unsqueeze(1)
#             sqrt_one_minus_alpha_hat_t = sqrt_one_minus_alpha_hat_t.unsqueeze(1)
#             beta_t =  beta_t.unsqueeze(1)
#
#         adjusted_noise = (beta_t / sqrt_one_minus_alpha_hat_t) * pred_noise
#         updated_x = (x - adjusted_noise) / sqrt_alpha_t
#
#         # 添加新的随机噪声项
#         z = torch.randn_like(x).to(device)
#         sigma_t = beta_t.sqrt()
#         updated_x = updated_x+ sigma_t * z
#
#         return updated_x
#
#     def generate_samples(self, model, x_init, num_steps = 100):
#         samples = x_init  # 使用原类别的数据初始化
#         for t in reversed(range(num_steps)):
#             samples = self.p_sample(model, samples, t)
#         return samples



if __name__ == '__main__':
    from models import ResidualFusion
    from models import transformer
    # 示例数据
    B, C, D = 4, 3, 256  # 批次大小、通道数和特征维度
    data = torch.randn((B, C, D)).to(device)
    #     # 示例缺失情况
    mask = torch.tensor([###输入数据mask为True表示该模态缺失，False表示有该模型
        [False, True, False],
        [True, False, False],
        [False, False, True],
        [True, True, False]
    ], dtype=torch.bool).to(device)

    # 定义模型
    model = transformer.Transformer_encoder(d_model=256, dim_feedforward=1024, num_encoder_layers=2).to(device)
    diffusion_model = DiffusionModel(beta_start=0.01, beta_end=0.2, num_steps=100)

    # fake_features = diffusion_model.generate_samples(model, data, missing_mask)  # 使用MLP生成数据
    # 前向传播示例
    # t = 10  # 示例时间步
    t = torch.randint(0, diffusion_model.num_steps, (data.shape[0],)).long()
    print("#",t)
    x_noisy, actual_noise= diffusion_model.q_sample(data, t, mask)#添加噪声,mask True部分
    # print(data,x_noisy,actual_noise)
    # 预测噪声
    pred_noise = model(x_noisy,t)#利用mask 为Flase的部分预测噪声，True的部分为0
    # print(data[0][:,:10], x_noisy[0][:,:10], pred_noise[0][:,:10])#torch.Size([4, 3, 256])

    #利用误差数据，重建原始目标数据
    fake_features = diffusion_model.generate_samples(model, x_noisy, mask)#去除噪声,mask True部分
    print(data,fake_features)

