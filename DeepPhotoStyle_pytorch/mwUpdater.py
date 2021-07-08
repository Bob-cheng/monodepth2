
class MaskWeightUpdater():
    def __init__(self, initweight, maskloss_thresh, upscaler=1.2,  downscaler=0.5, interval=10) -> None:
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
            if ref_value > self.maskloss_thresh * 1.5:
                self.mask_weight = self.init_weight
            elif ref_value < self.maskloss_thresh * 0.5:
                self.mask_weight = -self.init_weight
            else:
                self.mask_weight = (ref_value - self.maskloss_thresh) / self.maskloss_thresh * 2 * self.init_weight
                
            
        return self.mask_weight

    def get_mask_weight(self):
        return self.mask_weight