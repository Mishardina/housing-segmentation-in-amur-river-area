import torch
from catalyst import dl

class SegmentationCustomRunner(dl.SupervisedRunner):
    def handle_batch(self, batch):
        image, mask = batch['image'].float(), batch['mask']
        mask_pred = self.model(image)
        label_pred = torch.where(mask_pred > 0.5, 1, 0).flatten(0)
        label_target = mask.flatten(0)
        
        self.batch = {'input': image, 'mask_target': mask, 'mask_pred': mask_pred, 'label_target': label_target.unsqueeze(1), 'label_pred': label_pred.unsqueeze(1)}