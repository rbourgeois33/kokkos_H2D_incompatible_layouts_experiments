#include <Kokkos_Core.hpp>
#include <iostream>
#include "mynvtx.h"
#include <type_traits>
#include <string>

// Kokkos aliases
using Host = Kokkos::DefaultHostExecutionSpace;
using Device = Kokkos::DefaultExecutionSpace;

using LayoutRight = Kokkos::LayoutRight;
using LayoutLeft = Kokkos::LayoutLeft;

using ViewLLDevice = Kokkos::View<double **, LayoutLeft, Device>;
using ViewLRDevice = Kokkos::View<double **, LayoutRight, Device>;
using ViewLLHost = Kokkos::View<double **, LayoutLeft, Host>;
using ViewLRHost = Kokkos::View<double **, LayoutRight, Host>;

template <class Layout, class ExecSpace>
using View = Kokkos::View<double **, Layout, ExecSpace>;

// Wrappers to get nices names and colors in Nsight System
const std::string cpu_color = "cyan";
const std::string gpu_color = "green";
const std::string H2D_color = "yellow";
const std::string D2H_color = "purple";

template <class Layout, class ExecSpace>
std::string message_generator(const std::string &kernel_name)
{
  std::string message;

  if (std::is_same<Layout, LayoutLeft>::value && std::is_same<ExecSpace, Device>::value)
  {
    message = "LL Device";
  }
  else if (std::is_same<Layout, LayoutRight>::value && std::is_same<ExecSpace, Device>::value)
  {
    message = "LR Device";
  }
  else if (std::is_same<Layout, LayoutLeft>::value && std::is_same<ExecSpace, Host>::value)
  {
    message = "LL Host";
  }
  else if (std::is_same<Layout, LayoutRight>::value && std::is_same<ExecSpace, Host>::value)
  {
    message = "LR Host";
  }
  else
  {
    message = "ERROR";
  }

  // Prepend the kernel name followed by a space
  return kernel_name + " " + message;
}

template <class ExecSpace>
std::string color_generator()
{

  if (std::is_same<ExecSpace, Device>::value)
  {
    return gpu_color;
  }
  else if (std::is_same<ExecSpace, Host>::value)
  {
    return cpu_color;
  }
  else
  {
    return "ERROR";
  }
}

// kernel that we are timing
// Inputs are the layout, the execution space and the number of launches
template <class Layout, class ExecSpace>
void blurrKernel(View<Layout, ExecSpace> &view, const int nlaunch = 1)
{

  const int N0 = view.extent(0);
  const int N1 = view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({1, 1}, {N0 - 1, N1 - 1});

  auto message = message_generator<Layout, ExecSpace>(__func__);
  auto color = color_generator<ExecSpace>();

  Kokkos::fence();
  mynvtxRangePush(message, color);

  for (int i = 0; i < nlaunch; i++)
  {
    Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j) { view(i, j) = (1.0 / 5) * (view(i - 1, j) + view(i, j) + view(i + 1, j) + view(i, j - 1) + view(i, j + 1)); });
    Kokkos::fence();
  }

  mynvtxRangePop();
}

// kernel that we are timing
// Inputs are the layout, the execution space
template <class Layout, class ExecSpace>
void InitKernel(View<Layout, ExecSpace> &view, const double &value = 1.0)
{

  const int N0 = view.extent(0);
  const int N1 = view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({0, 0}, {N0, N1});

  auto message = message_generator<Layout, ExecSpace>(__func__);
  auto color = color_generator<ExecSpace>();

  Kokkos::fence();
  mynvtxRangePush(message, color);

  Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j) { view(i, j) = value + i - j; });

  Kokkos::fence();
  mynvtxRangePop();
}

template <class Layout_dest, class Layout_src, class ExecSpace>
void transposeKernel(View<Layout_dest, ExecSpace> &view_dest, View<Layout_src, ExecSpace> &view_src, const int nlaunch = 1)
{

  const int N0 = view_dest.extent(0);
  const int N1 = view_dest.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({0, 0}, {N0, N1});

  auto message = message_generator<Layout_dest, ExecSpace>(__func__);
  auto color = color_generator<ExecSpace>();

  Kokkos::fence();
  mynvtxRangePush(message, color);

  for (int i = 0; i < nlaunch; i++)
  {
    Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j) { view_dest(i, j) = view_src(i, j); });
    Kokkos::fence();
  }

  mynvtxRangePop();
}

// Kernel to check results

// Compare doubles
static const double delta = 100 * std::numeric_limits<double>::epsilon(); // Tolerance used for real number comparison
KOKKOS_INLINE_FUNCTION
bool is_equal(const double a, const double b)
{
  return (abs(a - b) < delta);
}

template <class Layout, class ExecSpace>
void check_result(View<Layout, ExecSpace> &view, const double &value = 1.0)
{

  const int N0 = view.extent(0);
  const int N1 = view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({0, 0}, {N0, N1});

  bool is_correct = true;

  // Perform the reduction
  Kokkos::parallel_reduce(
      "CheckValues", policy,
      KOKKOS_LAMBDA(const int i, const int j, bool &is_correct_local) {
        // Check if the value matches the expected pattern
        bool is_correct_thread = (is_equal(view(i, j), value + i - j));

        is_correct_local = is_correct_local && is_correct_thread;
      },
      Kokkos::LAnd<bool>(is_correct)); // Logical AND reduction

  Kokkos::fence();
  if (!is_correct)
  {
    throw std::runtime_error("View values do not match the expected pattern!");
  }
  else
  {
    std::cout << "Success ! The swapping deep copy worked" << std::endl;
  }
}

int main(int argc, char *argv[])
{

  // Initialize Kokkos runtime
  mynvtxRangePush("Kokkos::initialize", "red");
  Kokkos::initialize();
  mynvtxRangePop();

  // Main scope
  mynvtxRangePush("Main scope", "white");
  {

    // Size of the view
    const int ratio = 1;
    const int N0 = 30000 * ratio;
    const int N1 = 30000 / ratio;

    // Allocate the views
    mynvtxRangePush("alloc LL Device", gpu_color);
    auto device_view_LL = ViewLLDevice("device_view_LL", N0, N1);
    mynvtxRangePop();

    mynvtxRangePush("alloc LR Device", gpu_color);
    auto device_view_LR = ViewLRDevice("device_view_LR", N0, N1);
    mynvtxRangePop();

    mynvtxRangePush("alloc LL Host", cpu_color);
    auto host_view_LL = ViewLLHost("host_view_LL", N0, N1);
    mynvtxRangePop();

    mynvtxRangePush("alloc LR Host", cpu_color);
    auto host_view_LR = ViewLRHost("host_view_LR", N0, N1);
    mynvtxRangePop();

    // Launch init kernels
    InitKernel<LayoutLeft, Device>(device_view_LL, gpu_value);
    InitKernel<LayoutRight, Device>(device_view_LR, gpu_value);
    InitKernel<LayoutLeft, Host>(host_view_LL, cpu_value);
    InitKernel<LayoutRight, Host>(host_view_LR, cpu_value);

    // Launch blurr kernels
    const int nlaunch_gpu = 1;
    blurrKernel<LayoutLeft, Device>(device_view_LL, nlaunch_gpu);
    blurrKernel<LayoutRight, Device>(device_view_LR, nlaunch_gpu);

    const int nlaunch_cpu = 1;
    blurrKernel<LayoutLeft, Host>(host_view_LL, nlaunch_cpu);
    blurrKernel<LayoutRight, Host>(host_view_LR, nlaunch_cpu);

    // Deep copies
    mynvtxRangePush("deep copy H2D LL", H2D_color);
    Kokkos::deep_copy(device_view_LL, host_view_LL);
    mynvtxRangePop();

    mynvtxRangePush("deep copy H2D LR", H2D_color);
    Kokkos::deep_copy(device_view_LR, host_view_LR);
    mynvtxRangePop();

    mynvtxRangePush("deep copy D2H LL", D2H_color);
    Kokkos::deep_copy(host_view_LL, device_view_LL);
    mynvtxRangePop();

    mynvtxRangePush("deep copy D2H LR", D2H_color);
    Kokkos::deep_copy(host_view_LR, device_view_LR);
    mynvtxRangePop();


    const double cpu_value = 2.0;
    const double gpu_value = 4.0;
    // Deep copy LR to LL H2D transpose on device
    {
      mynvtxRangePush("check", "black");
      InitKernel<LayoutRight, Host>(host_view_LR, cpu_value);
      mynvtxRangePop();
      Kokkos::fence();

      mynvtxRangePush("deep copy H2D transpose on Device", H2D_color);

      mynvtxRangePush("allocate buffer", gpu_color);
      auto buffer_H2D = ViewLRDevice("buffer_H2D", N0, N1);
      mynvtxRangePop();

      mynvtxRangePush("deep copy H2D", H2D_color);
      Kokkos::deep_copy(buffer_H2D, host_view_LR);
      mynvtxRangePop();

      transposeKernel<LayoutLeft, LayoutRight, Device>(device_view_LL, buffer_H2D, nlaunch_gpu);
      mynvtxRangePop("deep copy H2D transpose on Device");

      mynvtxRangePush("check", "black");
      check_result<LayoutLeft, Device>(device_view_LL, cpu_value);
      mynvtxRangePop();
    }
    Kokkos::fence(); // Fence for out of scope dealloc of buffer

    // Deep copy LL to LR D2H, transpose on device
    {
      mynvtxRangePush("check", "black");
      InitKernel<LayoutLeft, Device>(device_view_LL, gpu_value);
      mynvtxRangePop();
      Kokkos::fence();

      mynvtxRangePush("deep copy D2H transpose on Device", D2H_color);

      mynvtxRangePush("allocate buffer", gpu_color);
      auto buffer_D2H = ViewLRDevice("buffer_D2H", N0, N1);
      mynvtxRangePop();

      transposeKernel<LayoutRight, LayoutLeft, Device>(buffer_D2H, device_view_LL, nlaunch_gpu);

      mynvtxRangePush("deep copy D2H", D2H_color);
      Kokkos::deep_copy(host_view_LR, buffer_D2H);
      mynvtxRangePop();

      mynvtxRangePop("deep copy D2H transpose on Device");

      mynvtxRangePush("check", "black");
      check_result<LayoutRight, Host>(host_view_LR, gpu_value);
      mynvtxRangePop();
    }
    Kokkos::fence(); // Fence for out of scope dealloc of buffer

    // Deep copy LR to LL H2D transpose on host
    {
      mynvtxRangePush("check", "black");
      InitKernel<LayoutRight, Host>(host_view_LR, cpu_value);
      mynvtxRangePop();
      Kokkos::fence();

      mynvtxRangePush("deep copy H2D transpose on Host", H2D_color);

      mynvtxRangePush("allocate buffer", cpu_color);
      auto buffer_H2D = ViewLLHost("buffer_H2D", N0, N1);
      mynvtxRangePop();

      transposeKernel<LayoutLeft, LayoutRight, Host>(buffer_H2D, host_view_LR, nlaunch_gpu);

      mynvtxRangePush("deep copy H2D", H2D_color);
      Kokkos::deep_copy(device_view_LL, buffer_H2D);
      mynvtxRangePop();

      mynvtxRangePop("deep copy H2D transpose on Host");

      mynvtxRangePush("check", "black");
      check_result<LayoutLeft, Device>(device_view_LL, cpu_value);
      mynvtxRangePop();
    }
    Kokkos::fence(); // Fence for out of scope dealloc of buffer

    // Deep copy LL to LR D2H
    {
      mynvtxRangePush("check", "black");
      InitKernel<LayoutLeft, Device>(device_view_LL, gpu_value);
      mynvtxRangePop();
      Kokkos::fence();

      mynvtxRangePush("deep copy D2H transpose on Host", D2H_color);

      mynvtxRangePush("allocate buffer", cpu_color);
      auto buffer_D2H = ViewLLHost("buffer_D2H", N0, N1);
      mynvtxRangePop();

      mynvtxRangePush("deep copy D2H", D2H_color);
      Kokkos::deep_copy(buffer_D2H, device_view_LL);
      mynvtxRangePop();

      transposeKernel<LayoutRight, LayoutLeft, Host>(host_view_LR, buffer_D2H, nlaunch_gpu);

      mynvtxRangePop("deep copy D2H transpose on Host");

      mynvtxRangePush("check", "black");
      check_result<LayoutRight, Host>(host_view_LR, gpu_value);
      mynvtxRangePop();
    }
    Kokkos::fence(); // Fence for out of scope dealloc of buffer
  }
  mynvtxRangePop();

  // Finalize Kokkos runtime
  mynvtxRangePush("Kokkos::finalize", "red");
  Kokkos::finalize();
  mynvtxRangePop();

  return 0;
}
