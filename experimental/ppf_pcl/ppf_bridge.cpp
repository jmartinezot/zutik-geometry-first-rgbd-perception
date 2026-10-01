// Experimental Drost-style PPF bridge.
//
// The executable deliberately has a small, auditable contract with Python:
// input coordinates and all distance thresholds are metres; the source is the
// CAD/model cloud; the target is the observed scene/candidate cloud; and every
// returned matrix maps source/model coordinates into target/camera coordinates
// (T_target_source == T_camera_object in the repository convention).

#include <pcl/features/ppf.h>
#include <pcl/point_types.h>
#include <pcl/registration/ppf_registration.h>

#include <Eigen/Core>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Point = pcl::PointNormal;
using Cloud = pcl::PointCloud<Point>;

struct Args {
  std::string model;
  std::string scene;
  std::string output;
  float angle_bin_deg = 10.0F;
  float distance_bin_m = 0.005F;
  unsigned int scene_reference_sampling_rate = 5U;
  float position_cluster_m = 0.03F;
  float rotation_cluster_deg = 25.0F;
  std::size_t top_k = 20U;
};

void usage() {
  std::cerr << "Usage: zutik_ppf_bridge --model MODEL.xyzn --scene SCENE.xyzn "
               "--output RESULT.json [options]\n"
               "  --angle-bin-deg D\n"
               "  --distance-bin-m M\n"
               "  --scene-reference-sampling-rate N\n"
               "  --position-cluster-m M\n"
               "  --rotation-cluster-deg D\n"
               "  --top-k N\n";
}

template <typename T>
T parse_value(const std::string& value, const char* option) {
  std::istringstream stream(value);
  T parsed{};
  if (!(stream >> parsed) || !stream.eof()) {
    throw std::runtime_error(std::string("invalid value for ") + option + ": " + value);
  }
  return parsed;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string option(argv[index]);
    if (option == "--help") {
      usage();
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + option);
    }
    const std::string value(argv[++index]);
    if (option == "--model") args.model = value;
    else if (option == "--scene") args.scene = value;
    else if (option == "--output") args.output = value;
    else if (option == "--angle-bin-deg") args.angle_bin_deg = parse_value<float>(value, option.c_str());
    else if (option == "--distance-bin-m") args.distance_bin_m = parse_value<float>(value, option.c_str());
    else if (option == "--scene-reference-sampling-rate") args.scene_reference_sampling_rate = parse_value<unsigned int>(value, option.c_str());
    else if (option == "--position-cluster-m") args.position_cluster_m = parse_value<float>(value, option.c_str());
    else if (option == "--rotation-cluster-deg") args.rotation_cluster_deg = parse_value<float>(value, option.c_str());
    else if (option == "--top-k") args.top_k = parse_value<std::size_t>(value, option.c_str());
    else throw std::runtime_error("unknown option: " + option);
  }
  if (args.model.empty() || args.scene.empty() || args.output.empty()) {
    throw std::runtime_error("--model, --scene, and --output are required");
  }
  if (args.angle_bin_deg <= 0.0F || args.distance_bin_m <= 0.0F ||
      args.scene_reference_sampling_rate == 0U || args.position_cluster_m <= 0.0F ||
      args.rotation_cluster_deg <= 0.0F || args.top_k == 0U) {
    throw std::runtime_error("PPF parameters must be positive");
  }
  return args;
}

Cloud::Ptr load_xyzn(const std::string& filename) {
  std::ifstream input(filename);
  if (!input) throw std::runtime_error("cannot open point cloud: " + filename);

  std::string magic;
  std::string units;
  std::size_t count = 0U;
  if (!(input >> magic >> units >> count) || magic != "ZUTIK_PPF_XYZ_NORMALS_V1" || units != "m") {
    throw std::runtime_error("invalid XYZN header; expected ZUTIK_PPF_XYZ_NORMALS_V1 m N");
  }
  if (count == 0U) throw std::runtime_error("point cloud is empty: " + filename);

  Cloud::Ptr cloud(new Cloud);
  cloud->reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    Point point;
    if (!(input >> point.x >> point.y >> point.z >> point.normal_x >> point.normal_y >> point.normal_z)) {
      throw std::runtime_error("truncated or malformed point cloud: " + filename);
    }
    const float normal_length = std::sqrt(point.normal_x * point.normal_x + point.normal_y * point.normal_y + point.normal_z * point.normal_z);
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) ||
        !std::isfinite(point.normal_x) || !std::isfinite(point.normal_y) || !std::isfinite(point.normal_z) ||
        normal_length <= 1e-6F) {
      throw std::runtime_error("non-finite or zero-length point normal: " + filename);
    }
    point.normal_x /= normal_length;
    point.normal_y /= normal_length;
    point.normal_z /= normal_length;
    cloud->push_back(point);
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1U;
  cloud->is_dense = true;
  return cloud;
}

void write_matrix(std::ostream& output, const Eigen::Affine3f& pose) {
  output << '[';
  const Eigen::Matrix4f matrix = pose.matrix();
  for (int row = 0; row < 4; ++row) {
    if (row != 0) output << ',';
    output << '[';
    for (int column = 0; column < 4; ++column) {
      if (column != 0) output << ',';
      output << std::setprecision(9) << matrix(row, column);
    }
    output << ']';
  }
  output << ']';
}

void write_failure(const std::string& filename, const std::string& message) {
  std::ofstream output(filename);
  output << "{\"status\":\"error\",\"error\":\"";
  for (const char character : message) {
    if (character == '\\' || character == '"') output << '\\';
    output << character;
  }
  output << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string output_filename;
  try {
    const Args args = parse_args(argc, argv);
    output_filename = args.output;
    const auto start = std::chrono::steady_clock::now();
    const Cloud::Ptr model = load_xyzn(args.model);
    const Cloud::Ptr scene = load_xyzn(args.scene);

    // PCL's reference implementation is the experimental algorithm under
    // test: all model pair PPFs are indexed, scene reference points are
    // sampled, and clustered pose votes are returned by PPFRegistration.
    pcl::PPFEstimation<Point, Point, pcl::PPFSignature> estimator;
    estimator.setInputCloud(model);
    estimator.setInputNormals(model);
    pcl::PointCloud<pcl::PPFSignature>::Ptr model_features(new pcl::PointCloud<pcl::PPFSignature>);
    estimator.compute(*model_features);

    const float angle_bin_rad = args.angle_bin_deg * static_cast<float>(M_PI) / 180.0F;
    pcl::PPFHashMapSearch::Ptr search(new pcl::PPFHashMapSearch(angle_bin_rad, args.distance_bin_m));
    search->setInputFeatureCloud(model_features);

    pcl::PPFRegistration<Point, Point> registration;
    registration.setSearchMethod(search);
    registration.setSceneReferencePointSamplingRate(args.scene_reference_sampling_rate);
    registration.setPositionClusteringThreshold(args.position_cluster_m);
    registration.setRotationClusteringThreshold(args.rotation_cluster_deg * static_cast<float>(M_PI) / 180.0F);
    registration.setInputSource(model);
    registration.setInputTarget(scene);
    Cloud aligned;
    registration.align(aligned);
    // Debian bookworm ships PCL 1.13.0.  Its public PPFRegistration API
    // exposes only the highest-vote final transformation; the ranked
    // getBestPoseCandidates() accessor was added in the newer API documented
    // by current PCL.  Do not reimplement private clustering here: emit the
    // official top-1 result and record this limitation in diagnostics.
    const Eigen::Affine3f final_pose(registration.getFinalTransformation());

    std::ofstream output(args.output);
    if (!output) throw std::runtime_error("cannot create result: " + args.output);
    output << "{\"status\":\"ok\",\"translation_units\":\"m\",\"transform_convention\":\"T_target_source\",\"model_points\":"
           << model->size() << ",\"scene_points\":" << scene->size() << ",\"runtime_s\":";
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    output << std::setprecision(9) << elapsed.count() << ",\"diagnostics\":{";
    output << "\"model_ppf_features\":" << model_features->size() << ",\"scene_reference_sampling_rate\":"
           << args.scene_reference_sampling_rate << ",\"converged\":" << (registration.hasConverged() ? "true" : "false")
           << ",\"pcl_version\":\"1.13.x\",\"ranked_pose_accessor_available\":false,\"requested_top_k\":" << args.top_k << "},\"hypotheses\":[";
    if (registration.hasConverged()) {
      output << "{\"rank\":1,\"votes\":null,\"transform\":";
      write_matrix(output, final_pose);
      output << '}';
    }
    output << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    if (!output_filename.empty()) {
      try { write_failure(output_filename, error.what()); } catch (...) { /* preserve stderr below */ }
    }
    std::cerr << "zutik_ppf_bridge: " << error.what() << '\n';
    return 2;
  }
}
