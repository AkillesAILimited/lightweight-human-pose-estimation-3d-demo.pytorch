// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>

namespace human_pose_estimation {

constexpr double pi = 3.14159265358979323846;

inline float get_alpha(float rate = 30.0, float cutoff = 1) {
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
inline std::vector<int> int_vector(const std::vector<T> & v) {
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

constexpr float eps = 1e-7; // np.spacing(1)

inline int get_similarity(const PoseCommon & common, const Pose &a, const Pose &b, float threashold=0.5) {
    int num_similar_kpt = 0;
    for (int kpt_id = 0; kpt_id < common.num_kpts; ++kpt_id) {
        float distance = 0.0;
        for (int i=0; i< int(a.keypoints[0].size()); ++i) {
            float dx = a.keypoints[kpt_id][i] - b.keypoints[kpt_id][i];
            float dx2 = dx*dx;
            distance += dx2;
        }
        float area = (a.bbox[1][0] - a.bbox[0][0])*(a.bbox[1][1] - a.bbox[0][1])*(a.bbox[1][2] - a.bbox[0][2]);
        float similarity = std::exp( - distance / (2.0 * (area + eps) * common.vars[kpt_id]));
        if (similarity > threashold) {
            ++ num_similar_kpt;
        }
    }

    return num_similar_kpt;
}

inline void propagate_ids(PoseCommon & common, std::vector<Pose> & previous_poses, std::vector<Pose> &current_poses, int threshold=3) {
    /* Propagate poses ids from previous frame results. Id is propagated,
    if there are at least `threshold` similar keypoints between pose from previous frame and current.

    :param previous_poses: poses from previous frame with ids
    :param current_poses: poses from current frame to assign ids
    :param threshold: minimal number of similar keypoints between poses
    :return: None
    */

    std::vector<int> current_poses_sorted_ids;
    for (int i=0; i< int(current_poses.size()); ++i) {
        current_poses_sorted_ids.push_back(i);
    }
    std::sort(current_poses_sorted_ids.begin(), current_poses_sorted_ids.end(), [&](int a, int b) {
        return current_poses[a].confidence > current_poses[b].confidence;
    });

    //if (current_poses.size() > 0) {
    //    assert(current_poses[0].confidence >= current_poses[1].confidence);
    //}

    std::vector<uint8_t> mask;
    for (int i=0; i< int(previous_poses.size()); ++i) {
        mask.push_back(1);
    }

    for (int current_pose_id_id = 0; current_pose_id_id < int(current_poses_sorted_ids.size()); ++current_pose_id_id){
        int current_pose_id = current_poses_sorted_ids[current_pose_id_id];
        int best_matched_id = -1; // None
        int best_matched_pose_id = -1; // None
        float best_matched_iou = 0;

        for (int previous_pose_id = 0; previous_pose_id < int(previous_poses.size()); ++ previous_pose_id) {
            if (mask[previous_pose_id] == 0)
                continue;
            float iou = get_similarity(common, current_poses[current_pose_id], previous_poses[previous_pose_id]);
            if (iou > best_matched_iou) {
                best_matched_iou = iou;
                best_matched_pose_id = previous_poses[previous_pose_id].id;
                best_matched_id = previous_pose_id;
            }
        }
        if (best_matched_iou >= threshold) {
            mask[best_matched_id] = 0;
        } else {
            best_matched_pose_id = -1;
        }
        current_poses[current_pose_id].update_id(common, best_matched_pose_id);
        if (best_matched_pose_id > -1) { //is not None:
            current_poses[current_pose_id].translation_filter = previous_poses[best_matched_id].translation_filter;
        }
    }
}

struct HumanPose {
    HumanPose(const std::vector<cv::Point3f>& keypoints = std::vector<cv::Point3f>(),
              const float& score = 0);

    std::vector<cv::Point3f> keypoints;
    float score;
};
} // namespace human_pose_estimation
