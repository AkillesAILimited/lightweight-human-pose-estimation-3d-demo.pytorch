#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <opencv2/core/core.hpp>

#include "extract_poses.hpp"
#include "parse_poses.hpp"

static std::vector<cv::Mat> wrap_feature_maps(PyArrayObject* py_feature_maps) {
    int num_channels = static_cast<int>(PyArray_SHAPE(py_feature_maps)[0]);
    int h = static_cast<int>(PyArray_SHAPE(py_feature_maps)[1]);
    int w = static_cast<int>(PyArray_SHAPE(py_feature_maps)[2]);
    float* data = static_cast<float*>(PyArray_DATA(py_feature_maps));
    std::vector<cv::Mat> feature_maps(num_channels);
    for (long c_id = 0; c_id < num_channels; c_id++)
    {
        feature_maps[c_id] = cv::Mat(h, w, CV_32FC1,
                                     data + c_id * PyArray_STRIDE(py_feature_maps, 0) / sizeof(float),
                                     PyArray_STRIDE(py_feature_maps, 1));
    }
    return feature_maps;
}

static PyObject* parse_poses_cpp(PyObject* self, PyObject* args) {

    // poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
    // parsed_poses parse_poses(const cv::Mat &features, const cv::Mat &heatmap, const cv::Mat &paf_map, float input_scale, int stride, float fx, bool is_video=false);
    // poses_3d, poses_2d = parse_poses_cpp(inference_result[0], inference_result[1], inference_result[2], input_scale, stride, fx, is_video)
    // (57, 32, 56) (19, 32, 56) (38, 32, 56) <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
    //    <class 'float'> <class 'int'> <class 'numpy.float32'> <class 'bool'>

    std::cerr << "parse_poses_cpp(...)" << std::endl << std::flush;
    PyArrayObject* py_feature_map = nullptr;
    PyArrayObject* py_heatmap = nullptr;
    PyArrayObject* py_paf_map = nullptr;

    float input_scale;
    int stride;
    float fx;
    int is_video;

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

    if (!PyArg_ParseTuple(args, "OOOfifp", &py_feature_map, &py_heatmap, &py_paf_map, &input_scale, &stride, &fx, &is_video))
    {
        throw std::runtime_error("passed non-numpy array as argument");
    }

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

    cv::Mat features;
    {
        int x = static_cast<int>(PyArray_SHAPE(py_feature_map)[0]);
        int y = static_cast<int>(PyArray_SHAPE(py_feature_map)[1]);
        int z = static_cast<int>(PyArray_SHAPE(py_feature_map)[2]);
        float* data = static_cast<float*>(PyArray_DATA(py_feature_map));
        std::cerr << x << "," << y << "," << z << std::endl << std::flush;
        int xyz[] = { x,y,z };
        features = cv::Mat(3, xyz, CV_32FC1, data);
    }

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

    cv::Mat heatmap;
    {
        int x = static_cast<int>(PyArray_SHAPE(py_heatmap)[0]);
        int y = static_cast<int>(PyArray_SHAPE(py_heatmap)[1]);
        int z = static_cast<int>(PyArray_SHAPE(py_heatmap)[2]);
        float* data = static_cast<float*>(PyArray_DATA(py_heatmap));
        std::cerr << x << "," << y << "," << z << std::endl << std::flush;
        int xyz[] = { x,y,z };
        heatmap = cv::Mat(3, xyz, CV_32FC1, data);
    }

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

    cv::Mat paf_map;
    {
        int x = static_cast<int>(PyArray_SHAPE(py_paf_map)[0]);
        int y = static_cast<int>(PyArray_SHAPE(py_paf_map)[1]);
        int z = static_cast<int>(PyArray_SHAPE(py_paf_map)[2]);
        float* data = static_cast<float*>(PyArray_DATA(py_paf_map));
        std::cerr << x << "," << y << "," << z << std::endl << std::flush;
        int xyz[] = { x,y,z };
        paf_map = cv::Mat(3, xyz, CV_32FC1, data);
    }

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

    //    feature_maps[c_id] = cv::Mat(h, w, CV_32FC1,
    //                                 data + c_id * PyArray_STRIDE(py_feature_maps, 0) / sizeof(float),
    //                                 PyArray_STRIDE(py_feature_maps, 1));

    auto rv = human_pose_estimation::parse_poses(features, heatmap, paf_map, input_scale, stride, fx, is_video!=0);

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

/*
    struct parsed_poses {
        //std::vector<cv::Point3f>
         std::vector<std::vector<float>>  translated_poses_3d;
        //std::vector<cv::Point3f> 
         std::vector<std::vector<float>>  poses_2d_scaled;

         // ??? format ???
    };
*/
    PyObject *pArray1 = nullptr;
    {
        float * out_data = (float*) malloc(sizeof(float) * rv.translated_poses_3d.size() *rv.translated_poses_3d[0].size());
        npy_intp dims[] = {static_cast<npy_intp>(rv.translated_poses_3d.size()),
                       static_cast<npy_intp>(rv.translated_poses_3d[0].size())};
        for (int i=0; i< int(rv.translated_poses_3d.size()); ++i) {
            for (int j=0; j < int(rv.translated_poses_3d[0].size()); ++j) {
                out_data[i * rv.translated_poses_3d[0].size() + j] = rv.translated_poses_3d[i][j];
            }
        }
        // PyObject 
        pArray1 = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, reinterpret_cast<void*>(out_data));
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(pArray1), NPY_ARRAY_OWNDATA);
    }
    //return Py_BuildValue("(N)", pArray);

    PyObject *pArray2 = nullptr;
    {
        float * out_data = (float*) malloc(sizeof(float) * rv.poses_2d_scaled.size() *rv.poses_2d_scaled[0].size());
        npy_intp dims[] = {static_cast<npy_intp>(rv.poses_2d_scaled.size()),
                       static_cast<npy_intp>(rv.poses_2d_scaled[0].size())};
        for (int i=0; i< int(rv.poses_2d_scaled.size()); ++i) {
            for (int j=0; j < int(rv.poses_2d_scaled[0].size()); ++j) {
                out_data[i * rv.poses_2d_scaled[0].size() + j] = rv.poses_2d_scaled[i][j];
            }
        }
        // PyObject 
        pArray2 = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, reinterpret_cast<void*>(out_data));
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(pArray2), NPY_ARRAY_OWNDATA);
    }
    //return Py_BuildValue("(N)", pArray);

    return Py_BuildValue("(NN)", pArray1, pArray2);
}

static PyObject* extract_poses(PyObject* self, PyObject* args) {
    PyArrayObject* py_heatmaps;
    PyArrayObject* py_pafs;
    int ratio;
    if (!PyArg_ParseTuple(args, "OOi", &py_heatmaps, &py_pafs, &ratio))
    {
        throw std::runtime_error("passed non-numpy array as argument");
    }
    std::vector<cv::Mat> heatmaps = wrap_feature_maps(py_heatmaps);
    std::vector<cv::Mat> pafs = wrap_feature_maps(py_pafs);

    std::vector<human_pose_estimation::HumanPose> poses = human_pose_estimation::extractPoses(
                heatmaps, pafs, ratio);

    size_t num_persons = poses.size();
    size_t num_keypoints = 0;
    if (num_persons > 0) {
        num_keypoints = poses[0].keypoints.size();
    }
    float* out_data = new float[num_persons * (num_keypoints * 3 + 1)];
    for (size_t person_id = 0; person_id < num_persons; person_id++)
    {
        for (size_t kpt_id = 0; kpt_id < num_keypoints * 3; kpt_id += 3)
        {
            out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 0] = poses[person_id].keypoints[kpt_id / 3].x;
            out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 1] = poses[person_id].keypoints[kpt_id / 3].y;
            out_data[person_id * (num_keypoints * 3 + 1) + kpt_id + 2] = poses[person_id].keypoints[kpt_id / 3].z;
        }
        out_data[person_id * (num_keypoints * 3 + 1) + num_keypoints * 3] = poses[person_id].score;
    }
    npy_intp dims[] = {static_cast<npy_intp>(num_persons),
                       static_cast<npy_intp>(num_keypoints * 3 + 1)};
    PyObject *pArray = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, reinterpret_cast<void*>(out_data));
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(pArray), NPY_ARRAY_OWNDATA);
    return Py_BuildValue("(N)", pArray);
}

PyMethodDef method_table[] = {
    {"extract_poses", static_cast<PyCFunction>(extract_poses), METH_VARARGS,
     "Extracts 2d poses from provided heatmaps and pafs"},
    {"parse_poses_cpp", static_cast<PyCFunction>(parse_poses_cpp), METH_VARARGS,
     "Parse 2d/3d poses from ..."},     
    {NULL, NULL, 0, NULL}
};

PyModuleDef pose_extractor_module = {
    PyModuleDef_HEAD_INIT,
    "pose_extractor",
    "Module for fast 2d pose extraction",
    -1,
    method_table
};

PyMODINIT_FUNC PyInit_pose_extractor(void) {
    PyObject* module = PyModule_Create(&pose_extractor_module);
    if (module == nullptr)
    {
        return nullptr;
    }
    import_array();
    if (PyErr_Occurred())
    {
        return nullptr;
    }

    return module;
}

/*

final res [[-3.42220873e-01 -2.37121940e+00  1.36021057e+02  8.69272649e-01
  -4.47139120e+00 -1.92274113e+01  1.19246094e+02  9.52342749e-01
  -3.61876988e+00  4.71134644e+01  1.37952316e+02 -1.00000000e+00
   1.28370228e+01 -4.49498081e+00  1.35466599e+02  7.91751921e-01
   2.98788338e+01  1.56848783e+01  1.29839371e+02  3.53191018e-01
   3.42746048e+01  1.83648682e+01  1.13704041e+02 -1.00000000e+00
   5.11018133e+00  4.83644638e+01  1.43420212e+02  3.87767524e-01
   3.45375824e+00  8.37497101e+01  1.50932877e+02 -1.00000000e+00
   2.58819008e+00  1.17640556e+02  1.61613510e+02 -1.00000000e+00
  -1.66531982e+01 -3.65018988e+00  1.32646896e+02  7.53126025e-01
  -2.53335857e+01  1.75410290e+01  1.28385269e+02  2.76258737e-01
  -8.07699966e+00  1.20446205e+01  1.10963585e+02  7.31388271e-01
  -1.54328365e+01  4.68879509e+01  1.48056763e+02  3.77475470e-01
  -1.97281837e+01  8.25580292e+01  1.48384064e+02 -1.00000000e+00
  -1.81924820e+01  1.16912971e+02  1.56312454e+02 -1.00000000e+00
  -1.29490721e+00 -1.94340038e+01  1.20895935e+02  7.65928507e-01
   5.03425694e+00 -1.60850449e+01  1.29250229e+02  9.59882200e-01
  -6.82711697e+00 -2.07387314e+01  1.19724846e+02 -1.00000000e+00
  -8.45976353e+00 -2.00736160e+01  1.27652199e+02  9.12060380e-01]
 [ 4.15071793e+01 -2.39552331e+00  1.18634430e+02  7.16541886e-01
   2.66106319e+01 -1.87869873e+01  1.10855186e+02  9.01975036e-01
   4.64751663e+01  4.46410789e+01  1.34551498e+02 -1.00000000e+00
   5.78569908e+01 -7.48637140e-01  1.13106613e+02  6.74003780e-01
   6.56605453e+01  2.32744160e+01  1.11329979e+02  6.69094741e-01
   4.68820457e+01  2.08195648e+01  9.73830948e+01  8.00002992e-01
   5.44429626e+01  4.68147507e+01  1.30964890e+02  3.02144229e-01
   5.54116745e+01  8.26766281e+01  1.35565781e+02 -1.00000000e+00
   5.94602203e+01  1.14086067e+02  1.50367172e+02 -1.00000000e+00
   2.99649448e+01 -3.90597582e+00  1.24441772e+02  6.09390080e-01
   2.80209160e+01  1.65992413e+01  1.37622025e+02  3.70783091e-01
   2.54709129e+01  2.68053818e+01  1.23966316e+02  6.09241545e-01
   4.25947647e+01  3.84810295e+01  1.51553909e+02  3.58791113e-01
   4.24071922e+01  7.16417847e+01  1.70222366e+02 -1.00000000e+00
   5.04723969e+01  1.02880379e+02  1.87895355e+02 -1.00000000e+00
   2.83301125e+01 -1.84934216e+01  1.08331627e+02  5.46823382e-01
   3.72160149e+01 -1.54193163e+01  1.08737343e+02  9.69643652e-01
   2.51201439e+01 -1.99956436e+01  1.10508087e+02 -1.00000000e+00
   3.23846092e+01 -1.98253593e+01  1.16438004e+02  9.18726325e-01]
 [-4.76369286e+01 -5.59190798e+00  1.35034119e+02  5.29984593e-01
  -3.32405968e+01 -2.49836578e+01  1.27744759e+02  8.34457397e-01
  -4.64240379e+01  4.59761467e+01  1.40603928e+02 -1.00000000e+00
  -4.62610474e+01 -7.20676327e+00  1.50877533e+02  3.10917825e-01
  -4.63634682e+01  1.66573124e+01  1.58130630e+02  1.26021847e-01
  -3.40014191e+01  1.20742817e+01  1.48463669e+02 -1.00000000e+00
  -4.63840485e+01  4.54336510e+01  1.50674042e+02 -1.00000000e+00
  -4.77163429e+01  8.31066895e+01  1.55956482e+02 -1.00000000e+00
  -5.51777992e+01  1.19370277e+02  1.60390656e+02 -1.00000000e+00
  -5.08068123e+01 -4.42839622e+00  1.19398216e+02  7.05469906e-01
  -5.09006348e+01  1.90612698e+01  1.15051094e+02  6.54418230e-01
  -3.55653114e+01  9.58122826e+00  1.19711479e+02 -1.00000000e+00
  -5.12835426e+01  4.47253036e+01  1.34984451e+02  1.25949800e-01
  -5.33966751e+01  8.12741089e+01  1.33536697e+02 -1.00000000e+00
  -5.47613869e+01  1.18455116e+02  1.33179504e+02 -1.00000000e+00
  -3.50991516e+01 -2.65641479e+01  1.29862076e+02  9.67542946e-01
  -4.05988884e+01 -2.41604061e+01  1.37885681e+02 -1.00000000e+00
  -3.31207275e+01 -2.51733685e+01  1.27971901e+02  9.74785030e-01
  -4.13575783e+01 -2.12625885e+01  1.22798439e+02 -1.00000000e+00]] [[ 4.0700000e+02  2.3200000e+02  8.6927265e-01  3.8500000e+02
   1.5100000e+02  9.5234275e-01 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  4.8400000e+02  2.3900000e+02  7.9175192e-01
   5.2400000e+02  2.7600000e+02  3.5319102e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00  4.2500000e+02  4.6600000e+02
   3.8776752e-01 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00  3.3000000e+02
   2.2100000e+02  7.5312603e-01  2.6500000e+02  3.0500000e+02
   2.7625874e-01  3.7400000e+02  2.9400000e+02  7.3138827e-01
   3.3800000e+02  4.5500000e+02  3.7747547e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  3.8500000e+02  1.3300000e+02  7.6592851e-01
   4.0700000e+02  1.4000000e+02  9.5988220e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00  4.5500000e+02  1.5100000e+02
   9.1206038e-01  1.9383759e+02]
 [ 6.4100000e+02  2.2800000e+02  7.1654189e-01  5.3100000e+02
   1.4400000e+02  9.0197504e-01 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  6.8900000e+02  2.2800000e+02  6.7400378e-01
   7.6200000e+02  3.6700000e+02  6.6909474e-01  7.0300000e+02
   3.5200000e+02  8.0000299e-01  7.0000000e+02  4.6600000e+02
   3.0214423e-01 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00  5.9700000e+02
   2.2400000e+02  6.0939008e-01  6.0800000e+02  3.1200000e+02
   3.7078309e-01  5.6100000e+02  3.5600000e+02  6.0924155e-01
   6.2300000e+02  4.6600000e+02  3.5879111e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  5.3100000e+02  1.2600000e+02  5.4682338e-01
   5.5000000e+02  1.2600000e+02  9.6964365e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00  6.0800000e+02  1.4400000e+02
   9.1872633e-01  2.0764166e+02]
 [ 1.6600000e+02  2.0600000e+02  5.2998459e-01  2.8300000e+02
   1.2600000e+02  8.3445740e-01 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  1.4800000e+02  2.0200000e+02  3.1091782e-01
   1.9100000e+02  3.0500000e+02  1.2602185e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00  1.8800000e+02
   2.0600000e+02  7.0546991e-01  1.9900000e+02  3.0500000e+02
   6.5441823e-01 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
   1.5100000e+02  4.5100000e+02  1.2594980e-01 -1.0000000e+00
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  2.6800000e+02  1.1100000e+02  9.6754295e-01
  -1.0000000e+00 -1.0000000e+00 -1.0000000e+00  2.1700000e+02
   1.2600000e+02  9.7478503e-01 -1.0000000e+00 -1.0000000e+00
  -1.0000000e+00  8.0159622e+01]]



*/