
class MaskWeightUpdater():
    def __init__(self, initweight, maskloss_thresh, init_ratio, total_steps, upscaler=1.5,  downscaler=0.7, interval=20) -> None:
        self.upscaler = upscaler
        self.downscaler = downscaler
        self.interval = interval
        self.init_weight = initweight
        self.init_ratio = init_ratio
        self.total_steps = total_steps
        self.current_step = 0
        self.maskloss_thresh  = maskloss_thresh
        self.mask_weight = initweight
    
    def get_target_ratio(self):
        end_step = 0.6 * self.total_steps
        slope = (self.init_ratio - self.maskloss_thresh) / end_step
        if self.current_step > end_step:
            return self.maskloss_thresh
        else:
            return self.init_ratio - slope * self.current_step

    def get_target_ratio2(self):
        m = self.init_ratio
        f = 0.6 * self.total_steps
        g = self.maskloss_thresh
        x = self.current_step
        if self.current_step > f:
            return self.maskloss_thresh
        else:
            return (m-g)/(f*f)*x*x + 2 * (g-m)/f * x + m

    
    def step(self, mask_ratio):
        ref_value = mask_ratio
        self.current_step += 1
        if self.current_step % self.interval == 0:
            # if maskloss < self.maskloss_thresh:
            #     self.mask_weight *= self.downscaler
            # else:
            #     self.mask_weight *= self.upscaler

            # self.mask_weight = self.init_weight

            ## try to fix the mask area to maskloss_thresh
            # bound = 2.0
            # if (ref_value > self.maskloss_thresh * bound):
            #     self.mask_weight = self.init_weight
            # elif (ref_value < self.maskloss_thresh / bound):
            #     self.mask_weight = - self.init_weight
            # elif (ref_value >= self.maskloss_thresh):
            #     self.mask_weight =  (ref_value / self.maskloss_thresh -1)  / (bound - 1) * self.init_weight
            # else:
            #     self.mask_weight = - (self.maskloss_thresh / ref_value - 1)  / (bound - 1) * self.init_weight

            # target_ratio = self.get_target_ratio()
            target_ratio = self.get_target_ratio2() # square
            print("target_ratio:", target_ratio, "current_ratio", ref_value)
            if ref_value >= target_ratio:
                self.mask_weight *= self.upscaler
            else:
                self.mask_weight *= self.downscaler

        return self.mask_weight

    def get_mask_weight(self):
        return self.mask_weight