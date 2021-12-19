#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include <opencv2/core/core.hpp>

//#include "extract_poses.hpp"
//#include "parse_poses.hpp"
#include "human_poses.hpp"

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

std::vector<human_pose_estimation::Pose> previous_poses_2d;
human_pose_estimation::PoseCommon common;

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

    auto rv = human_pose_estimation::parse_poses(previous_poses_2d, common, features, heatmap, paf_map, input_scale, stride, fx, is_video!=0);

    std::cerr << __FILE__ << ":" << __LINE__ << std::endl << std::flush;

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

