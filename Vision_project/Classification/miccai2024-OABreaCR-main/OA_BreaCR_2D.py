import torch
import torch.nn as nn
from module_2D import AttentionPool, SpatialTransformer, AttentionAlign
from ResNet2D import RN18_extrator, RN18_Classifer

class OA_BreaCR(nn.Module):

    def __init__(self, in_channels=64, size=[32, 32]):
        super(OA_BreaCR, self).__init__()
        
        self.encoder =  RN18_extrator(n_input_channels=3)
        self.attention_pool = AttentionPool(in_channels=in_channels)
        self.align = AttentionAlign(in_planes=2, out_planes=2, stride=1)
        self.spatial_transformer = SpatialTransformer(size=size)
        self.predictor_cur_prior = RN18_Classifer(in_planes=64, num_classes=3)
        self.predictor_future = RN18_Classifer(in_planes=192, num_classes=3)

    def forward(self, cur, prior):
        f_cur = self.encoder(cur)
        f_prior = self.encoder(prior)
        
        a_cur = self.attention_pool(f_cur)
        a_prior = self.attention_pool(f_prior)
        
        pred_cur = self.predictor_cur_prior(f_cur)
        pred_prior = self.predictor_cur_prior(f_prior)

        a = torch.cat([a_cur, a_prior], dim=1)
        flow = self.align(a)
        print("flow shape:", flow.shape)
        
        f_prior_hat = self.spatial_transformer(f_prior, flow)
        f_diff = f_cur - f_prior_hat
        # print("f_diff shape:", f_diff.shape)
        
        f_all = torch.cat([f_diff, f_cur, f_prior_hat], dim=1)
        pred_future = self.predictor_future(f_all)
        # print("pred_future shape", pred_future.shape)

        return pred_cur, pred_prior, pred_future 
    
if __name__ == "__main__":
    t1_tensor = torch.randn(4, 3, 128, 128)
    t2_tensor = torch.randn(4, 3, 128, 128)
    model = OA_BreaCR(size=[32, 32])
    output = model(t1_tensor, t2_tensor)
    