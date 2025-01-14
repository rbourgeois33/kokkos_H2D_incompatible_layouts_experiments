#include <Kokkos_Core.hpp>
#include <iostream>
#include "mynvtx.h"
#include <type_traits>
#include <string>

using _TYPE_ = float;

// Kokkos aliases
using Host = Kokkos::DefaultHostExecutionSpace;
using Device = Kokkos::DefaultExecutionSpace;

using LayoutRight = Kokkos::LayoutRight;
using LayoutLeft = Kokkos::LayoutLeft;

using ViewLLDevice = Kokkos::View<_TYPE_ **, LayoutLeft, Device>;
using ViewLRDevice = Kokkos::View<_TYPE_ **, LayoutRight, Device>;
using ViewLLHost = Kokkos::View<_TYPE_ **, LayoutLeft, Host>;
using ViewLRHost = Kokkos::View<_TYPE_ **, LayoutRight, Host>;

template <class Layout, class ExecSpace>
using View = Kokkos::View<_TYPE_ **, Layout, ExecSpace>;

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
void InitKernel(View<Layout, ExecSpace> &view, const _TYPE_ &value = 1.0, const bool range=true)
{

  const int N0 = view.extent(0);
  const int N1 = view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({0, 0}, {N0, N1});

  auto message = message_generator<Layout, ExecSpace>(__func__);
  auto color = color_generator<ExecSpace>();

  Kokkos::fence();
  if (range) mynvtxRangePush(message, color);

  Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j) { view(i, j) = value + i - j; });

  Kokkos::fence();
  if (range) mynvtxRangePop();
}

// kernel that we are timing
// Inputs are the layout, the execution space
template <class Layout, class ExecSpace>
void ReadKernel(const View<Layout, ExecSpace> &view)
{

  const int N0 = view.extent(0);
  const int N1 = view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({0, 0}, {N0, N1});

  auto message = message_generator<Layout, ExecSpace>(__func__);
  auto color = color_generator<ExecSpace>();

  Kokkos::fence();
  mynvtxRangePush(message, color);

  Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j) { _TYPE_ a = view(i,j); });

  Kokkos::fence();
  mynvtxRangePop();
}

template <class Layout_dest, class Layout_src, class ExecSpace>
void transposeKernel(View<Layout_dest, ExecSpace> &view_dest, const View<Layout_src, ExecSpace> &view_src, const int nlaunch = 1)
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

// Compare _TYPE_s
static const _TYPE_ delta = 100 * std::numeric_limits<_TYPE_>::epsilon(); // Tolerance used for real number comparison
KOKKOS_INLINE_FUNCTION
bool is_equal(const _TYPE_ a, const _TYPE_ b)
{
  return (abs(a - b) < delta);
}

template <class Layout, class ExecSpace>
void check_result(View<Layout, ExecSpace> &view, const _TYPE_ &value = 1.0)
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

const _TYPE_ cpu_value = 2.0;
const _TYPE_ gpu_value = 4.0;

//deep copy routine that handles non compatible exec spaces and layouts
//Non trivial template parameters:
//ExecSpace_tmp: the execution space on which we allocate the tmp buffer and perform the "transpose"
//Non trivial runtime parameters:
//use_transpose_kernel: Use the transpose kernel i wrote to perform the transpose. Otherwise, use deep_copy
//check: Check the validity of the result
template <class Layout_dest, class Layout_src, class ExecSpace_dest, class ExecSpace_src, class ExecSpace_tmp=Device>
void deep_copy_generalized(View<Layout_dest, ExecSpace_dest> &view_dest,
                           View<Layout_src, ExecSpace_src> &view_src,
                           const bool use_transpose_kernel=false,
                           const bool check=false)
{

  if (check)
    {
      const bool srcIsDevice = std::is_same<ExecSpace_src, Device>::value;
      const _TYPE_ value_init = srcIsDevice ? gpu_value : cpu_value;
      //mynvtxRangePush("check", "black");
      InitKernel<Layout_src, ExecSpace_src>(view_src, value_init, /*create a nvtx range*/ false);
      Kokkos::fence();
      //mynvtxRangePop();
    }

  std::string string_Layout_src = (std::is_same<LayoutLeft, Layout_src>::value) ? "LL" : "LR";
  std::string string_Layout_dest = (std::is_same<LayoutLeft, Layout_dest>::value) ? "LL" : "LR";
  std::string string_Space_src = (std::is_same<ExecSpace_src, Device>::value) ? "D" : "H";
  std::string string_Space_dest = (std::is_same<ExecSpace_dest, Device>::value) ? "D" : "H";
  std::string string_transpose_type = (use_transpose_kernel) ? "kernel" : "deep copy";
  std::string string_transpose_space = (std::is_same<ExecSpace_tmp, Device>::value) ? "D" : "H";

  std::string name ="deep_copy gen. "
     // string_Layout_src+"2"+string_Layout_dest+" "
      +string_Space_src+"2"+string_Space_dest
      +" with "+string_transpose_type +" on "+ string_transpose_space;

  std::string name_alloc="allocation on "+string_transpose_space;
  std::string name_transpose="transpose via "+string_transpose_type+ " on "+string_transpose_space;
  std::string name_deep_copy="deep copy "+string_Space_src+"2"+string_Space_dest;

  mynvtxRangePush(name);

  constexpr bool sameExecSpace = (std::is_same<ExecSpace_dest, ExecSpace_src>::value);
  constexpr bool transposeOnSrcExecSpace = (std::is_same<ExecSpace_src, ExecSpace_tmp>::value);
  constexpr bool transposeOnDestExecSpace = (std::is_same<ExecSpace_dest, ExecSpace_tmp>::value);

  const int N0 = view_src.extent(0);
  const int N1 = view_src.extent(1);

  //Same Exespace --> Deep copy works even for different layouts
  if constexpr (sameExecSpace)
    {
      Kokkos::deep_copy(view_dest, view_src);
    }
  else
    {
      if constexpr (transposeOnSrcExecSpace)
        {
          mynvtxRangePush(name_alloc);
          auto view_tmp = View<Layout_dest, ExecSpace_tmp>("view_tmp", N0,N1);
          mynvtxRangePop(name_alloc);

          mynvtxRangePush(name_transpose);
          if(use_transpose_kernel)
            {
              transposeKernel<Layout_dest, Layout_src, ExecSpace_tmp>(view_tmp, view_src);
            }
          else //use deep copy (kokkos doc recommandation)
            {
              Kokkos::deep_copy(view_tmp, view_src); //Legal because on the same execspace, does the transpose
            }
          mynvtxRangePop(name_transpose);

          mynvtxRangePush(name_deep_copy);
          Kokkos::deep_copy(view_dest, view_tmp);//Legal because same layout, does the H2D / D2H
          mynvtxRangePop(name_deep_copy);

        }
      else if constexpr (transposeOnDestExecSpace)
        {
          mynvtxRangePush(name_alloc);
          auto view_tmp = View<Layout_src, ExecSpace_tmp>("view_tmp", N0,N1);
          mynvtxRangePop(name_alloc);

          mynvtxRangePush(name_deep_copy);
          Kokkos::deep_copy(view_tmp, view_src); //Legal because same layout, does the H2D / D2H
          mynvtxRangePop(name_deep_copy);

          mynvtxRangePush(name_transpose);
          if (use_transpose_kernel)
            {
              transposeKernel<Layout_dest, Layout_src, ExecSpace_tmp>(view_dest, view_tmp);
            }
          else//use deep copy (kokkos doc recommandation)
            {
              Kokkos::deep_copy(view_dest, view_tmp); //Legal because on the same execspace, does the transpose
            }
          mynvtxRangePop(name_transpose);
        }
      else
        {
          throw("Error");
        }
    }
  Kokkos::fence();
  mynvtxRangePop(name);

  if (check)
  {
  const bool srcIsDevice = std::is_same<ExecSpace_src, Device>::value;
  const _TYPE_ value_init = srcIsDevice ? gpu_value : cpu_value;
 // mynvtxRangePush("check", "black");
  check_result<Layout_dest, ExecSpace_dest>(view_dest, value_init);
  //mynvtxRangePop();
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
    const int N0 = 3000000;
    const int N1 = 60;

    std::cout<<"Arrays size= "<<float(N0*N1*sizeof(_TYPE_))/(1024*1024*1024)<<"GB\n";

    // Size of the view
    //const int N0 = 2450;
    //const int N1 = 2450;

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
    //InitKernel<LayoutRight, Host>(host_view_LR, cpu_value);

    // Read kernels
//    ReadKernel<LayoutLeft, Device>(device_view_LL);
//    ReadKernel<LayoutRight, Device>(device_view_LR);
//    ReadKernel<LayoutLeft, Host>(host_view_LL);
//    ReadKernel<LayoutRight, Host>(host_view_LR);

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


    //Deep copy generalized: cases that it reduce to simple deep copies
    //Deep copies generalized - same layout diff execspace
//    deep_copy_generalized<LayoutLeft, LayoutLeft, Device, Host>(device_view_LL, host_view_LL, false, true);
//    deep_copy_generalized<LayoutLeft, LayoutLeft, Host, Device>(host_view_LL, device_view_LL, false, true);
//    deep_copy_generalized<LayoutRight, LayoutRight, Device, Host>(device_view_LR, host_view_LR, false, true);
//    deep_copy_generalized<LayoutRight, LayoutRight, Host, Device>(host_view_LR, device_view_LR, false, true);
//    //Deep copies generalized - same execspace diff Layout
//    deep_copy_generalized<LayoutLeft, LayoutRight, Device, Device>(device_view_LL, device_view_LR, false, true);
//    deep_copy_generalized<LayoutRight, LayoutLeft, Device, Device>(device_view_LR, device_view_LL, false, true);
//    deep_copy_generalized<LayoutLeft, LayoutRight, Host, Host>(host_view_LL, host_view_LR, false, true);
//    deep_copy_generalized<LayoutRight, LayoutLeft, Host, Host>(host_view_LR, host_view_LL, false, true);

    //Deep copy generalized: diff layout and diff execspace H2D, use deep copy for transpose, on H/D
    deep_copy_generalized<LayoutLeft, LayoutRight, Device, Host, Host>(device_view_LL, host_view_LR, /* use kernel */ false , true);
    deep_copy_generalized<LayoutLeft, LayoutRight, Device, Host, Device>(device_view_LL, host_view_LR, /* use kernel */false , true);
    //Deep copy generalized: diff layout and diff execspace H2D, use kernel for transpose on H/D
    deep_copy_generalized<LayoutLeft, LayoutRight, Device, Host, Host>(device_view_LL, host_view_LR, /* use kernel */ true , true);
    deep_copy_generalized<LayoutLeft, LayoutRight, Device, Host, Device>(device_view_LL, host_view_LR, /* use kernel */ true , true);


    //Deep copy generalized: diff layout and diff execspace D2H, use deep copy for transpose, on H/D
    deep_copy_generalized<LayoutRight, LayoutLeft, Host, Device, Host>(host_view_LR,device_view_LL, /* use kernel */ false, true);
    deep_copy_generalized<LayoutRight, LayoutLeft, Host, Device, Device>(host_view_LR,device_view_LL, /* use kernel */ false , true);
    //Deep copy generalized: diff layout and diff execspace D2H, use kernel for transpose, on H/D
    deep_copy_generalized<LayoutRight, LayoutLeft, Host, Device, Host>(host_view_LR,device_view_LL, /* use kernel */ true, true);
    deep_copy_generalized<LayoutRight, LayoutLeft, Host, Device, Device>(host_view_LR,device_view_LL, /* use kernel */ true , true);
  }
  mynvtxRangePop("Main scope");

  // Finalize Kokkos runtime
  mynvtxRangePush("Kokkos::finalize", "red");
  Kokkos::finalize();
  mynvtxRangePop();

  return 0;
}
