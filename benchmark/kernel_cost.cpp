// Measures the cost of one evaluation of each FMM kernel on a pair of points relative to that of
// Kernel<Biharmonic3D<Dim>> and prints the invocations of POLATORY_KERNEL_COST.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <polatory/fmm/gradient_kernel.hpp>
#include <polatory/fmm/gradient_transpose_kernel.hpp>
#include <polatory/fmm/hessian_kernel.hpp>
#include <polatory/fmm/kernel.hpp>
#include <polatory/rbf/cov_exponential.hpp>
#include <polatory/rbf/cov_gaussian.hpp>
#include <polatory/rbf/cov_generalized_cauchy3.hpp>
#include <polatory/rbf/cov_generalized_cauchy5.hpp>
#include <polatory/rbf/cov_generalized_cauchy7.hpp>
#include <polatory/rbf/cov_generalized_cauchy9.hpp>
#include <polatory/rbf/cov_spheroidal3.hpp>
#include <polatory/rbf/cov_spheroidal5.hpp>
#include <polatory/rbf/cov_spheroidal7.hpp>
#include <polatory/rbf/cov_spheroidal9.hpp>
#include <polatory/rbf/polyharmonic_even.hpp>
#include <polatory/rbf/polyharmonic_odd.hpp>
#include <polatory/types.hpp>
#include <random>
#include <scalfmm/container/point.hpp>
#include <string>
#include <vector>

using polatory::Index;
using polatory::fmm::GradientKernel;
using polatory::fmm::GradientTransposeKernel;
using polatory::fmm::HessianKernel;
using polatory::fmm::Kernel;
using polatory::rbf::internal::Biharmonic2D;
using polatory::rbf::internal::Biharmonic3D;
using polatory::rbf::internal::CovExponential;
using polatory::rbf::internal::CovGaussian;
using polatory::rbf::internal::CovGeneralizedCauchy3;
using polatory::rbf::internal::CovGeneralizedCauchy5;
using polatory::rbf::internal::CovGeneralizedCauchy7;
using polatory::rbf::internal::CovGeneralizedCauchy9;
using polatory::rbf::internal::CovSpheroidal3FastPart;
using polatory::rbf::internal::CovSpheroidal5FastPart;
using polatory::rbf::internal::CovSpheroidal7FastPart;
using polatory::rbf::internal::CovSpheroidal9FastPart;
using polatory::rbf::internal::Triharmonic2D;
using polatory::rbf::internal::Triharmonic3D;

namespace {

constexpr int kRounds = 5;

// Evaluations per second on cache-resident pairs of random points in the unit box, with each
// thread working on its own share, as ScalFMM's near field is compute-bound.
template <class Kernel>
double measure_throughput(const Kernel& kernel) {
  constexpr int kDim = Kernel::kDim;
  constexpr Index kNumPairs = 1 << 15;
  constexpr int kRepeats = 512;
  using Position = scalfmm::container::point<double, kDim>;
  std::vector<Position> xs(kNumPairs);
  std::vector<Position> ys(kNumPairs);
  std::mt19937 gen;
  std::uniform_real_distribution<double> dist(-0.5, 0.5);
  for (Index i = 0; i < kNumPairs; i++) {
    for (auto k = 0; k < kDim; k++) {
      xs.at(i).at(k) = dist(gen);
      ys.at(i).at(k) = dist(gen);
    }
  }

  double sum{};
  auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel reduction(+ : sum)
  {
    auto n_threads = omp_get_num_threads();
    auto id = omp_get_thread_num();
    auto begin = kNumPairs * id / n_threads;
    auto end = kNumPairs * (id + 1) / n_threads;
    for (auto r = 0; r < kRepeats; r++) {
      for (auto i = begin; i < end; i++) {
        for (auto v : kernel.evaluate(xs.at(i), ys.at(i))) {
          sum += v;
        }
      }
    }
  }
  auto t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  volatile double sink = sum;
  static_cast<void>(sink);
  return static_cast<double>(kRepeats) * kNumPairs / t;
}

// The first rounds in a process run on a cold machine, so the rounds are interleaved with those
// of the reference and the best of each is kept.
template <class Kernel, class RefKernel>
double relative_cost(const Kernel& kernel, const RefKernel& ref_kernel) {
  auto ref_throughput = 0.0;
  auto throughput = 0.0;
  for (auto r = 0; r < kRounds; r++) {
    ref_throughput = std::max(ref_throughput, measure_throughput(ref_kernel));
    throughput = std::max(throughput, measure_throughput(kernel));
  }
  return ref_throughput / throughput;
}

template <template <int> class Rbf, int Dim>
void measure(const std::string& rbf_name, const std::vector<double>& params) {
  Biharmonic3D<Dim> ref_rbf({1.0});
  Kernel<Biharmonic3D<Dim>> ref_kernel(ref_rbf);
  Rbf<Dim> rbf(params);

  std::cout << "POLATORY_KERNEL_COST(" << rbf_name << "<" << Dim << ">, "
            << relative_cost(Kernel<Rbf<Dim>>(rbf), ref_kernel) << ", "
            << relative_cost(GradientKernel<Rbf<Dim>>(rbf), ref_kernel) << ", "
            << relative_cost(GradientTransposeKernel<Rbf<Dim>>(rbf), ref_kernel) << ", "
            << relative_cost(HessianKernel<Rbf<Dim>>(rbf), ref_kernel) << ")\n";
}

template <template <int> class Rbf>
void measure(const std::string& rbf_name, const std::vector<double>& params) {
  measure<Rbf, 1>(rbf_name, params);
  measure<Rbf, 2>(rbf_name, params);
  measure<Rbf, 3>(rbf_name, params);
}

}  // namespace

int main() {
  std::cout.precision(3);
  std::cerr << "threads: " << omp_get_max_threads() << '\n';

  std::vector<double> polyharmonic{1.0};
  std::vector<double> covariance{1.0, 1.0};
  measure<Biharmonic2D>("Biharmonic2D", polyharmonic);
  measure<Biharmonic3D>("Biharmonic3D", polyharmonic);
  measure<CovExponential>("CovExponential", covariance);
  measure<CovGaussian>("CovGaussian", covariance);
  measure<CovGeneralizedCauchy3>("CovGeneralizedCauchy3", covariance);
  measure<CovGeneralizedCauchy5>("CovGeneralizedCauchy5", covariance);
  measure<CovGeneralizedCauchy7>("CovGeneralizedCauchy7", covariance);
  measure<CovGeneralizedCauchy9>("CovGeneralizedCauchy9", covariance);
  measure<CovSpheroidal3FastPart>("CovSpheroidal3FastPart", covariance);
  measure<CovSpheroidal5FastPart>("CovSpheroidal5FastPart", covariance);
  measure<CovSpheroidal7FastPart>("CovSpheroidal7FastPart", covariance);
  measure<CovSpheroidal9FastPart>("CovSpheroidal9FastPart", covariance);
  measure<Triharmonic2D>("Triharmonic2D", polyharmonic);
  measure<Triharmonic3D>("Triharmonic3D", polyharmonic);
}
