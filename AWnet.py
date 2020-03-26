import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import PWCNet

class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,ker_size,stri,pad):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel,out_channel, 3, 1, 1)

    def forward(self,x):
        return self.conv2(F.relu(self.conv1(x)))

class ada_mask(nn.Module):
    def __init__(self,input_channel):
        super(ada_mask, self).__init__()
        self.mask_head = nn.Conv2d(input_channel, 64, 3, 1, 1)
        self.mask_Res1 = ResBlock(64, 64, 3, 1, 1)
        self.mask_Res2 = ResBlock(64, 64, 3, 1, 1)
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1)
        self.mask_Res1_1d = ResBlock(128, 128, 3, 1, 1)
        self.mask_Res1_2d = ResBlock(128, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.mask_Res2_1d = ResBlock(256, 256, 3, 1, 1)
        self.mask_Res2_2d = ResBlock(256, 256, 3, 1, 1)
        self.down3 = nn.Conv2d(256, 512, 3, 2, 1)
        self.mask_Res3_1d = ResBlock(512, 512, 3, 1, 1)
        self.mask_Res3_2d = ResBlock(512, 512, 3, 1, 1)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_Res3_1u = ResBlock(512, 256, 3, 1, 1)
        self.mask_Res3_2u = ResBlock(256, 256, 3, 1, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_Res2_1u = ResBlock(256, 128, 3, 1, 1)
        self.mask_Res2_2u = ResBlock(128, 128, 3, 1, 1)
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.mask_Res1_1u = ResBlock(128, 64, 3, 1, 1)
        self.mask_Res1_2u = ResBlock(64, 64, 3, 1, 1)
        self.mask_tail = nn.Conv2d(64, 26, 3, 1, 1)

    def forward(self,input):
        maskd0 = self.mask_Res2(self.mask_Res1(self.mask_head(input)))  # scale = 1
        maskd1 = self.mask_Res1_2d(self.mask_Res1_1d(self.down1(maskd0)))  # scale = 1/2
        maskd2 = self.mask_Res2_2d(self.mask_Res2_1d(self.down2(maskd1)))  # scale = 1/4
        maskd3 = self.mask_Res3_2d(self.mask_Res3_1d(self.down3(maskd2)))   # scale = 1/8
        masku2 = self.mask_Res3_2u(self.mask_Res3_1u(self.up3(maskd3)))+maskd2     # scale = 1/4
        masku1 = self.mask_Res2_2u(self.mask_Res2_1u(self.up2(masku2)))+maskd1     # scale = 1/2
        masku0 = self.mask_Res1_2u(self.mask_Res1_1u(self.up1(masku1)))+maskd0      #scale = 1
        mask = self.mask_tail(masku0)
        return mask

def warp( x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

   # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask,mask

class pwc_residual(nn.Module):
    def __init__(self):
        super(pwc_residual, self).__init__()
        self.pad = nn.ReplicationPad2d(2)
        self.FlowNet = PWCNet.PWCDCNet()
        self.mask = ada_mask(11)
        # for param in self.FlowNet.parameters():
        #     param.requires_grad = False

    def forward(self,ref,sr):
        [b,c,h,w] = ref.size()

        # sr_x = F.avg_pool2d(sr,kernel_size=2,stride=2)
        # ref_x = F.avg_pool2d(ref,kernel_size=2,stride=2)
        # flow = self.FlowNet(torch.cat((sr_x,ref_x),1))
        # flow = F.upsample(flow,scale_factor=4*2,mode='bilinear',align_corners=False)*20*2

        # sr_x = F.avg_pool2d(sr,kernel_size=4,stride=4)
        # ref_x = F.avg_pool2d(ref,kernel_size=4,stride=4)
        # flow = self.FlowNet(torch.cat((sr_x,ref_x),1))
        # flow = F.upsample(flow,scale_factor=4*4,mode='bilinear',align_corners=False)*20*4

        flow = self.FlowNet(torch.cat((sr,ref),1))
        flow = F.upsample(flow,scale_factor=4,mode='bilinear',align_corners=False)*20
        #warp_blurdetect,mask_blur = warp(ref_blurdetect, flow.contiguous())
        warp_ref,mask_ref = warp(ref,flow.contiguous())
        ref_structure = torch.zeros([b,c,5*5,h,w]).cuda()
        ref_padding = self.pad(warp_ref)
        for i in range(5):
            for j in range(5):
                ref_structure[:,:,i*5+j,:,:] = ref_padding[:,:,i:i+h,j:j+w]
        warp_ref_stru = ref_structure.view(b,c*5*5,h,w)
        # print(sr.shape)
        # print(sr_noise.shape)
        features = torch.cat((warp_ref,sr,flow,(warp_ref-sr)),1)
        mask = self.mask(features.detach())
        mask_sigmoid = F.sigmoid(mask[:,25,:,:])*mask_ref[:,0,:,:]
        ref_r = torch.sum(warp_ref_stru[:,0:25,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
        ref_g = torch.sum(warp_ref_stru[:,25:50,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
        ref_b = torch.sum(warp_ref_stru[:,50:75,:,:]*mask[:,0:25,:,:],1)*mask_sigmoid
        ref_contribution = torch.stack([ref_r,ref_g,ref_b],1)
        sr_contribution = ( 1- mask_sigmoid ).unsqueeze(1)
        return ref_contribution + sr*sr_contribution, warp_ref, mask