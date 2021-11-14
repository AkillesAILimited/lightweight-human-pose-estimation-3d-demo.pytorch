// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "human_pose.hpp"

namespace human_pose_estimation {

constexpr double pi = 3.14159265358979323846;

float get_alpha(float rate = 30.0, float cutoff = 1) {
  float tau = 1.0 / (2.0 * pi * cutoff);
  float te = 1.0 / rate;
  return 1.0 / (1.0 + tau / te);
}

struct LowPassFilter {
  bool has_previous = false;
  float previous = 0;

  inline float operator ()(float x, float alpha = 0.5) {
    if (!has_previous) {
      has_previous = true;
      previous = x;
      return x;
    }
    float x_filtered = alpha * x + (1.0 - alpha) * previous;
    previous = x_filtered;
    return x_filtered;
  }
};

struct OneEuroFilter {
  float freq;
  float mincutoff;
  float beta;
  float dcutoff;
  LowPassFilter filter_x;
  LowPassFilter filter_dx;
  bool has_previous;
  float previous;
  float dx;

  void init(float freq_ = 15, float mincutoff_ = 1, float beta_ = 1, float dcutoff_=1) {
    freq = freq_;
    mincutoff = mincutoff_;
    beta = beta_;
    dcutoff = dcutoff_;
    has_previous = false;
    previous = 0;
    dx = 0;
  }

  float operator()(float x) {
    if (!has_previous) {
      dx = 0;
      has_previous = true;
    } else {
      dx = (x - previous) * freq;
    }
    float dx_smoothed = filter_dx(dx, get_alpha(freq, dcutoff));
    float cutoff = mincutoff + beta * abs(dx_smoothed);
    float x_filtered = filter_x(x, get_alpha(freq, cutoff));
    previous = x;
    return x_filtered;
  }
};

struct PoseCommon {
    int num_kpts = 18;
    std::vector<std::string> kpt_names {"neck", "nose",
                 "l_sho", "l_elb", "l_wri", "l_hip", "l_knee", "l_ank",
                 "r_sho", "r_elb", "r_wri", "r_hip", "r_knee", "r_ank",
                 "r_eye", "l_eye",
                 "r_ear", "l_ear"};
    std::vector<float> sigmas { 0.079, 0.026, 0.079, 0.072, 0.062, 0.107, 0.087, 0.089, 0.079, 0.072, 0.062, 0.107, 0.087, 0.089, 0.025, 0.025, 0.035, 0.035 };
    std::vector<float> vars;
    int last_id = -1;
    int color[3] {0, 224, 255};

    PoseCommon() {
      for (auto &s: sigmas) {
        vars.push_back((s*2.0)*(s*2.0));
      }
    }
};

template<typename T>
std::vector<int> int_vector(const std::vector<T> & v) {
  std::vector<int> v2;
  v2.resize(v.size());
  for (int i=0; i< int(v.size()); ++i) {
    v2[i] = int(v[i]);
  }
  return v2;
}

struct Pose {


    std::vector<std::vector<float>> keypoints;
    std::vector<float> confidence;

    std::vector<std::vector<float>> bbox;
    std::vector<OneEuroFilter> translation_filter;
    int id = -1; // -1 is None for us...
   
    void init(const std::vector<std::vector<float>> &keypoints_, const std::vector<float> &confidence_) {

      keypoints = keypoints_;
      confidence = confidence_;

      std::vector<std::vector<int>> found_keypoints;
      found_keypoints.resize(2);
      int nkeypoints = 0;
      for (int n=0; n<int(keypoints.size()); ++n) {
        if (keypoints[n][0] != -1) {
          ++nkeypoints;
        }
      }
      found_keypoints.resize(nkeypoints);
      int found_kpt_id = 0;
      for (int kpt_id=0; kpt_id < int(keypoints.size()); ++kpt_id) {
        if (keypoints[kpt_id][0] == -1) {
          continue;
        }
        found_keypoints[found_kpt_id] = int_vector(keypoints[kpt_id]);
        ++ found_kpt_id;
      }
      // make a bounding rect
      std::vector<float> maxes;
      maxes.resize(keypoints[0].size());
      for (int i=0; i< int(maxes.size()); ++i) {
        maxes[i] = -1e12;
      }
      std::vector<float> mines;
      mines.resize(keypoints[0].size());
      for (int i=0; i< int(mines.size()); ++i) {
        mines[i] = 1e12;
      }

      for (int i=0; i< int(keypoints.size()); ++i) {
        for (int j =0; j < int(keypoints[0].size()); ++j) {
          if (keypoints[i][j] > maxes[j]) {
            maxes[j] = keypoints[i][j];
          }
          if (keypoints[i][j] < mines[j]) {
            mines[j] = keypoints[i][j];
          }
        }
      }

      bbox = std::vector<std::vector<float>> { maxes, mines };
      id = -1;
      translation_filter.resize(3);
      translation_filter[0].init(80, 1.0, 0.01);
      translation_filter[1].init(80, 1.0, 0.01);
      translation_filter[2].init(80, 1.0, 0.01);
    }

    void update_id(PoseCommon &pc, int id_ = -1) {
      id = id_;
      if (id == -1) {
        id = pc.last_id + 1;
        pc.last_id += 1;
      }
    }

    std::vector<float> filter(const std::vector<float> &f) {
      std::vector<float> rv;
      for (int i=0; i< int(translation_filter.size()); ++i) {
        rv.push_back(translation_filter[i](f[i]));
      }
      return rv;
    }
};


/* TODO ...

def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses_sorted_ids = list(range(len(current_poses)))
    current_poses_sorted_ids = sorted(
        current_poses_sorted_ids, key=lambda pose_id: current_poses[pose_id].confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in current_poses_sorted_ids:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)
        if best_matched_pose_id is not None:
            current_poses[current_pose_id].translation_filter = previous_poses[best_matched_id].translation_filter

*/

HumanPose::HumanPose(const std::vector<cv::Point3f>& keypoints,
                     const float& score)
    : keypoints(keypoints),
      score(score) {}
} // namespace human_pose_estimation

/*

import cv2
import numpy as np

from modules.one_euro_filter import OneEuroFilter


class Pose:
    num_kpts = 18
    kpt_names = ['neck', 'nose',
                 'l_sho', 'l_elb', 'l_wri', 'l_hip', 'l_knee', 'l_ank',
                 'r_sho', 'r_elb', 'r_wri', 'r_hip', 'r_knee', 'r_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
    sigmas = np.array([.79, .26, .79, .72, .62, 1.07, .87, .89, .79, .72, .62, 1.07, .87, .89, .25, .25, .35, .35],
                      dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [0, 224, 255]

    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        found_keypoints = np.zeros((np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32)
        found_kpt_id = 0
        for kpt_id in range(keypoints.shape[0]):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        self.bbox = cv2.boundingRect(found_keypoints)
        self.id = None
        self.translation_filter = [OneEuroFilter(freq=80, beta=0.01),
                                   OneEuroFilter(freq=80, beta=0.01),
                                   OneEuroFilter(freq=80, beta=0.01)]

    def update_id(self, id=None):
        self.id = id
        if self.id is None:
            self.id = Pose.last_id + 1
            Pose.last_id += 1

    def filter(self, translation): ###### <======
        filtered_translation = []
        for coordinate_id in range(3):
            filtered_translation.append(self.translation_filter[coordinate_id](translation[coordinate_id]))
        return filtered_translation


def get_similarity(a, b, threshold=0.5):
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(-distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id]))
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt


def propagate_ids(previous_poses, current_poses, threshold=3):
    """Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    """
    current_poses_sorted_ids = list(range(len(current_poses)))
    current_poses_sorted_ids = sorted(
        current_poses_sorted_ids, key=lambda pose_id: current_poses[pose_id].confidence, reverse=True)  # match confident poses first
    mask = np.ones(len(previous_poses), dtype=np.int32)
    for current_pose_id in current_poses_sorted_ids:
        best_matched_id = None
        best_matched_pose_id = None
        best_matched_iou = 0
        for previous_pose_id in range(len(previous_poses)):
            if not mask[previous_pose_id]:
                continue
            iou = get_similarity(current_poses[current_pose_id], previous_poses[previous_pose_id])
            if iou > best_matched_iou:
                best_matched_iou = iou
                best_matched_pose_id = previous_poses[previous_pose_id].id
                best_matched_id = previous_pose_id
        if best_matched_iou >= threshold:
            mask[best_matched_id] = 0
        else:  # pose not similar to any previous
            best_matched_pose_id = None
        current_poses[current_pose_id].update_id(best_matched_pose_id)
        if best_matched_pose_id is not None:
            current_poses[current_pose_id].translation_filter = previous_poses[best_matched_id].translation_filter



*/