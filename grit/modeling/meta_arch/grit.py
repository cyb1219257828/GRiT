from typing import Dict, List, Optional, Tuple
import torch
from detectron2.config import configurable
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class GRiT(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        **kwargs):
        super().__init__(**kwargs)
        assert self.proposal_generator is not None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training
        assert detected_instances is None

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(features, proposals)
        if do_postprocess:
            assert not torch.jit.is_scripting(), \
                "Scripting is not supported for postprocess."
            return GRiT._postprocess(
                results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        targets_task = batched_inputs[0]['task']
        for anno_per_image in batched_inputs:
            assert targets_task == anno_per_image['task']

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        proposals, roihead_textdecoder_losses = self.roi_heads(
            features, proposals, gt_instances, targets_task=targets_task)

        losses = {}
        losses.update(roihead_textdecoder_losses)
        losses.update(proposal_losses)

        return losses

    def _show_image(self, images, gt_instances, features):
        '''        
        # "id": 1, "name": "Pedestrian"
        # "id": 2, "name": "Cyclist"
        # "id": 3, "name": "Vehicle"
        # "id": 4, "name": "Infrastructure"
        # "id": 5, "name": "N/A"
        '''
        import numpy as np
        import cv2
        image0 = np.transpose(images[0].cpu().numpy(), (1,2,0))
        # float16 转 uint8
        scaled = (image0 - np.min(image0)) / (np.max(image0) - np.min(image0)) * 255
        image0 = np.uint8(scaled)
        caption = gt_instances[0]._fields['gt_object_descriptions'].data[0]
        categorie = gt_instances[0]._fields['gt_classes'].cpu().numpy()
        bbox = gt_instances[0]._fields['gt_boxes'].tensor.cpu().numpy()[0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        image0 = image0.copy()
        cv2.rectangle(image0, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(image0, caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(image0, f'categorie={categorie}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imwrite('/data1/yubo/code/GRiT-master/test/image0.png', image0)

        for p in range(3, 8):
            # feature0_p3 = np.transpose(features['p3'][0].cpu().detach_().numpy(), (1,2,0))
            # scaled = (feature0_p3 - np.min(feature0_p3)) / (np.max(feature0_p3) - np.min(feature0_p3)) * 255
            # feature0_p3 = np.uint8(scaled)
            # _, _, dims_p3 = feature0_p3.shape

            feature = np.transpose(features['p'+str(p)][0].cpu().detach_().numpy(), (1,2,0))
            scaled = (feature - np.min(feature)) / (np.max(feature) - np.min(feature)) * 255
            feature = np.uint8(scaled)
            _, _, dims_p = feature.shape
            for i in range(dims_p):
                cv2.imwrite('/data1/yubo/code/GRiT-master/test/feature0_p' + str(p) + '/feature0_p'  + str(p) +  '_' + str(i) + '.png', feature[:, :, i])
        
    def _show_proposal(self, images, proposals, gt_instances, num):
        import numpy as np
        import cv2
        img0 = np.transpose(images[0].cpu().numpy(), (1,2,0))
        # float16 转 uint8
        scaled = (img0 - np.min(img0)) / (np.max(img0) - np.min(img0)) * 255
        img0 = np.uint8(scaled)
        # img0 = img0.astype(np.uint8)
        proposal1 = proposals[0]._fields['proposal_boxes'].tensor.cpu().detach_().numpy()
        # float32 转 uint8
        # proposal1 = (proposal1 * 255).astype(np.uint8)
        # proposal1 = proposal1.astype(np.int8)
        dims_proposal1, _ = proposal1.shape
        # img0 = cv2.resize(img0, (128, 128))
        img0_first10 = img0.copy()
        img0_first100 = img0.copy()
        img0_all = img0.copy()
        for i in range(10):
            x1, y1, x2, y2 = int(proposal1[i][0]), int(proposal1[i][1]), int(proposal1[i][2]), int(proposal1[i][3])
            cv2.rectangle(img0_first10, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.imwrite('/data1/yubo/code/GRiT-master/test/proposal/proposal_stage' + str(num) + '_first10.png', img0_first10)
        for i in range(100):
            x1, y1, x2, y2 = int(proposal1[i][0]), int(proposal1[i][1]), int(proposal1[i][2]), int(proposal1[i][3])
            cv2.rectangle(img0_first100, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.imwrite('/data1/yubo/code/GRiT-master/test/proposal/proposal_stage' + str(num) + '_first100.png', img0_first100)
        for i in range(dims_proposal1):
            x1, y1, x2, y2 = int(proposal1[i][0]), int(proposal1[i][1]), int(proposal1[i][2]), int(proposal1[i][3])
            cv2.rectangle(img0_all, (x1, y1), (x2, y2), (255, 255, 0), 1)
        cv2.imwrite('/data1/yubo/code/GRiT-master/test/proposal/proposal_stage' + str(num) + '_all.png', img0_all)