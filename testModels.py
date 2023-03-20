import torch
import torch.nn as nn
import torch.nn.functional as F

# class LocalSeModule(nn.Module):
#     def __init__(self, in_size,local_size=3, reduction=4):
#         super(LocalSeModule, self).__init__()
#         self.local_size=local_size
#         self.lse = nn.Sequential(
#             nn.AdaptiveAvgPool2d(local_size),
#             nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_size // reduction),
#
#             nn.PReLU(in_size//reduction),
#             nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_size),
#             nn.PReLU(in_size)
#         )
#
#     def forward(self, x):
#         print(x.shape)
#         unavgpool=nn.AdaptiveAvgPool2d(self.local_size)
#         print(F.softmax(self.lse(x), 1).shape)
#         print(unavgpool(F.softmax(self.lse(x), 1).shape))
#         # return torch.mul(x,unavgpool(F.softmax(self.lse(x), 1)))
#         # print(F.softmax(self.lse(x),dim=1))
#         # y=self.lse(x).repeat(1,1,3,3)
#         # print(y.shape)
#         # print(self.lse(x).shape)
#         # return torch.mul(x, self.lse(x).expand_as(x))
#         # return x * self.se(x) #.expand_as(x)
#
#
# arr=torch.randn(1,2,4,4)
#
# lse=LocalSeModule(2,2,1)
#
#
#
# # print(arr)
# print(lse(arr))
# #
# # test_num=torch.tensor([-1.4986,1.4986])
# #
# # print((arr[:,:,1,1]))
# # print(F.softmax(arr[:,:,1,1],dim=1))
#
# # print(F.softmax(test_num))
# x=torch.randn(1,2,2,2)
# print(x)
# print(nn.AdaptiveAvgPool2d(4)(x))
#
# #
# # print(x.view(1,5*2*2,1,1))
# # print(x.view(1,5*2*2,1,1).expand(1,4,2,2))
# # print(x.view(1,4,1,1).expand(1,4,2,2))
# # print(x.view(1,4,1,1).expand(1,4,2,2))
# # print(x.repeat(2,2,1,1).view(1,1,4,4).transpose(1,3).view(1,1,4,4))


# ---------------------------- SE ---------------------------------
class SE(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------------- C3LSE ---------------------------------
class LSEBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=16,local_size=5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.lse = nn.Sequential(
            nn.AdaptiveAvgPool2d(local_size),
            nn.Conv2d(c1,c_, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_),

            nn.PReLU(c_),
            nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.PReLU(c2)
        )

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        b, c, m, n = x.size()

        att = self.lse(x)
        att = F.softmax(att,1)  #
        att = F.adaptive_avg_pool2d(att,[m,n])
        out = x1*att

        return x + out if self.add else out


class C3LSE(C3):
    # C3 module with SEBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(LSEBottleneck(c_, c_, shortcut) for _ in range(n)))




class LocalSE(nn.Module):
    def __init__(self, in_size,local_size=3, reduction=4):
        super(LocalSE, self).__init__()
        self.lse = nn.Sequential(
            # nn.AvgPool2d(local_size,local_size),
            nn.AdaptiveAvgPool2d(local_size),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),

            nn.PReLU(in_size//reduction),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.PReLU(in_size)
        )

    def forward(self, x):
        b,c,m,n = x.shape
        r = x
        att = self.lse(x)
        att = F.softmax(att,1)  #
        att = F.adaptive_avg_pool2d(att,[m,n])
        x = r*att
        return x






# input_arr = torch.rand(1,1,64,64)
# lse_att = LocalSeModule(1,4,1)
# output_arr = lse_att(input_arr)
# print(output_arr.shape)

# c = torch.rand(1,2,4,4)
# net = LocalSeModule(2,2,1)
# c1 = net(c)
# print(c)
# print(c1)
# c = torch.rand(1,2,4,4)
# net = LocalSeModule(1,2,1)
# c1 = net(c)
# print(c)
# print(c1)



# class LocalSeModule(nn.Module):
#     def __init__(self, in_size,local_size=4, reduction=4):
#         super(LocalSeModule, self).__init__()
#         self.lse = nn.Sequential(
#             nn.AdaptiveAvgPool2d(local_size),
#             nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_size // reduction),
#
#             nn.PReLU(in_size//reduction),
#             nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(in_size),
#             nn.PReLU(in_size)
#         )
#
#
#
#     def forward(self, x):
#         b,c,m,n = x.shape
#         r = x
#         att = self.lse(x)
#         att = F.softmax(att,1)
#         print(att)
#         att = F.adaptive_avg_pool2d(att,[m,n])
#         print(att)
#         x = r*att
#
#         return x
c = torch.rand(1,2,4,4)
net = LocalSE(2,2,1)
c1 = net(c)
# print(c)
print(c1)