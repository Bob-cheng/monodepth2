
class MaskWeightUpdater():
    def __init__(self, upscaler, downscaler, interval, initweight, maskloss_thresh) -> None:
        self.upscaler = upscaler
        self.downscaler = downscaler
        self.interval = interval
        self.init_weight = initweight
        self.current_step = 0
        self.maskloss_thresh  = maskloss_thresh
        self.mask_weight = initweight
    
    def step(self, maskloss):
        self.current_step += 1
        if self.current_step % self.interval == 0:
            if maskloss < self.maskloss_thresh:
                self.mask_weight *= self.downscaler
            else:
                self.mask_weight *= self.upscaler
        return self.mask_weight

    def get_mask_weight(self):
        return self.mask_weight