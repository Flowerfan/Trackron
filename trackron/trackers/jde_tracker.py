import time
from collections import deque

import numpy as np
import cv2
import torch
from trackron.structures.tracklet import (BaseTrack, TrackState,
                                          joint_tracklets,
                                          remove_duplicate_tracklets,
                                          sub_tracklets)
from trackron.data.utils import get_size_with_aspect_ratio

from .utils.matching import iou_distance, embedding_distance, linear_assignment, fuse_motion
from .build import TRACKER_REGISTRY
from .base_tracker import PostProcess
from .utils.kalman_filter import KalmanFilter
from .siamese_tracker import SiameseTracker
from .utils.preprocessing import numpy_to_torch


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (
                1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_num):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_num == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_num = frame_num
        self.start_frame = frame_num

    def re_activate(self, new_track, frame_num, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_num = frame_num
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_num, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_num: int
        :type update_feature: bool
        :return:
        """
        self.frame_num = frame_num
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame,
                                      self.end_frame)


@TRACKER_REGISTRY.register()
class JDETracker(SiameseTracker):

    def setup(self):
        ### MOT parameters
        super().setup()
        self.num_classes = self.cfg.MODEL.NUM_CLASS
        self.max_time_lost = 30
        self.max_age = 30
        self.score_thresh = self.det_thresh

        self.resize = (608, 1088)
        self.post_process = PostProcess(image_sz=(self.resize[0] // 4,
                                                  self.resize[1] // 4),
                                        box_fmt='xyxy',
                                        box_absolute=True)

    def reset_mot(self, image):
        ### parameter
        self.ori_image_szs = torch.tensor([image.shape[:2]],
                                          dtype=torch.float32,
                                          device=self.device)
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.last_detect_tracks = []
        self.kalman_filter = KalmanFilter()
        self.mot_ref_info = {}

    def init_mot(self, image, info: dict, category=None) -> dict:
        # Time initialization
        tic = time.time()
        self.reset_mot(image)
        self.category = category
        return self.track_mot(image, info)

    def track_mot(self, image, info):
        resize_image = cv2.resize(image, self.resize[::-1])
        im_tensor = numpy_to_torch(resize_image).to(self.device) / 255.0
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.net.track_mot(im_tensor)
        det_results = self.post_process(output, self.ori_image_szs)[0]
        dets = torch.cat(
            [det_results['boxes'], det_results['scores'][:, None]], dim=1)
        remain_inds = dets[:, 4] > self.det_thresh
        dets = dets[remain_inds].cpu().numpy()
        id_feature = det_results['embs'][remain_inds].cpu().numpy()

        track_results = self.update_mot(dets, id_feature)
        return track_results, det_results

    def update_mot(self, dets, id_feature=None):
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30)
                for (tlbrs, f) in zip(dets[:, :5], id_feature)
            ]
        else:
            detections = []
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        ''' Step 2: First association, with embedding'''
        strack_pool = joint_tracklets(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
        #strack.predict()
        STrack.multi_predict(strack_pool)
        dists = embedding_distance(strack_pool, detections)
        #dists = iou_distance(strack_pool, detections)
        dists = fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_num)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_num, new_id=False)
                refind_stracks.append(track)
        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [
            strack_pool[i] for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_num)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_num, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists,
                                                                thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_num)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_num)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_num - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_tracklets(self.tracked_stracks,
                                               activated_starcks)
        self.tracked_stracks = joint_tracklets(self.tracked_stracks,
                                               refind_stracks)
        self.lost_stracks = sub_tracklets(self.lost_stracks,
                                          self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_tracklets(self.lost_stracks,
                                          self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_tracklets(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [{
            'tracking_id': track.track_id,
            'bbox': track.tlbr,
            'score': track.score,
            'active': 1.0
        } for track in self.tracked_stracks if track.is_activated]

        # logger.debug('===========Frame {}=========='.format(self.frame_num))
        # logger.debug('Activated: {}'.format(
        #     [track.track_id for track in activated_starcks]))
        # logger.debug('Refind: {}'.format(
        #     [track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format(
        #     [track.track_id for track in removed_stracks]))

        return output_stracks
