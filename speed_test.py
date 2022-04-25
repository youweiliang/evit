import time
import torch
from timm import models
from torchprofile import profile_macs


all_models = \
"""
swin_small_patch4_window7_224
swin_tiny_patch4_window7_224
__Twins__
tnt_s_patch16_224
tnt_b_patch16_224
__Twins__
twins_pcpvt_small
twins_pcpvt_base
twins_svt_small
twins_svt_base
__visformer__
visformer_tiny
visformer_small
__vit/deit__
deit_small_patch16_224
deit_base_patch16_224
deit_base_patch16_384
__CaiT__
cait_xxs24_224
cait_xxs24_384
cait_xxs36_224
cait_xxs36_384
cait_s24_224
__CoaT__
coat_tiny
coat_mini
coat_lite_tiny
coat_lite_mini
coat_lite_small
__ConvViT__

convit_small
convit_base

__NFNet__
nfnet_f0
__EfficientNet__
efficientnet_b0
efficientnet_b1
efficientnet_b2
"""


def speed_test(model, ntest=50, batchsize=128, x=None, **kwargs):
    """
    Model should be in cuda!
    """
    if x is None:
        x = torch.rand(batchsize, 3, 224, 224).cuda()
    else:
        batchsize = x.shape[0]
    model.eval()

    start = time.time()
    for i in range(ntest):
        model(x, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    elapse = end - start
    speed = batchsize * ntest / elapse
    return speed


def test(model):
    model = model.cuda()
    base_bz = 32
    for img_size in range(224, 384+16, 16):
        try:
            x = torch.rand(4, 3, img_size, img_size).cuda()
            model(x)
            print("found image size:", img_size)
            break
        except:
            continue
    for e in range(1, 9):
        try:
            x = torch.rand(base_bz*(2**e), 3, img_size, img_size).cuda()
            model(x)
        except RuntimeError as error:
            # print("error msg:", error)
            max_bz = base_bz*(2**(e - 1))
            print("max batch size:", max_bz)
            break
        finally:
            del x
    x = torch.rand(max_bz, 3, img_size, img_size).cuda()
    speed = speed_test(model, x=x)
    print("1st speed:", speed)
    speed = speed_test(model, x=x)
    print("2nd speed:", speed)
    # speed = speed_test(model, x=x)
    # print("speed:", speed)
    x = torch.rand(1, 3, img_size, img_size).cuda()
    try:
        macs = profile_macs(model, x) * 1e-9
    except Exception as e:
        print("compute MAC error")
        macs = 0
    return int(speed), macs


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    log = ""
    for line in all_models.split("\n"):
        if line.startswith('_') or line == '':
            continue
        model_name = line.split(' ')[0]
        model_func = models.__getattribute__(model_name)
        model = model_func()
        print("\ncreated model", model_name)
        n_params = num_params(model) * 1e-6  # million
        print("num_params(M):", n_params)
        speed, macs = test(model)
        print("MACs(G):", macs)
        log += f"{model_name}\t{n_params}\t{speed}\t{macs}\n"
        del model
    print("finished test\nLog:")
    print(log)


if __name__ == "__main__":
    main()
