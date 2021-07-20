
class MaskWeightUpdater():
    def __init__(self, initweight, maskloss_thresh, upscaler=1.2,  downscaler=0.5, interval=30) -> None:
        self.upscaler = upscaler
        self.downscaler = downscaler
        self.interval = interval
        self.init_weight = initweight
        self.current_step = 0
        self.maskloss_thresh  = maskloss_thresh
        self.mask_weight = initweight
    
    def step(self, maskloss):
        ref_value = maskloss.item()
        self.current_step += 1
        if self.current_step % self.interval == 0:
            # if maskloss < self.maskloss_thresh:
            #     self.mask_weight *= self.downscaler
            # else:
            #     self.mask_weight *= self.upscaler

            # self.mask_weight = self.init_weight

            ## try to fix the mask area to maskloss_thresh
            bound = 2.0
            if (ref_value > self.maskloss_thresh * bound):
                self.mask_weight = self.init_weight
            elif (ref_value < self.maskloss_thresh / bound):
                self.mask_weight = -self.init_weight
            elif (ref_value >= self.maskloss_thresh):
                self.mask_weight = (ref_value / self.maskloss_thresh - 1) / (bound - 1) * self.init_weight
            else:
                self.mask_weight =  - (self.maskloss_thresh / ref_value - 1)  / (bound - 1) * self.init_weight
                
        return self.mask_weight

    def get_mask_weight(self):
        return self.mask_weight