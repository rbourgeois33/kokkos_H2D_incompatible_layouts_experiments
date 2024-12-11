#include <Kokkos_Core.hpp>
#include <iostream>
#include "mynvtx.h"
#include <type_traits>
#include <string>

//Kokkos aliases
using Host=Kokkos::DefaultHostExecutionSpace;
using Device=Kokkos::DefaultExecutionSpace;

using LayoutRight=Kokkos::LayoutRight;
using LayoutLeft=Kokkos::LayoutLeft;

using ViewLLDevice = Kokkos::View<double**, LayoutLeft, Device>;
using ViewLRDevice = Kokkos::View<double**, LayoutRight, Device>;
using ViewLLHost = Kokkos::View<double**, LayoutLeft, Host>;
using ViewLRHost = Kokkos::View<double**, LayoutRight, Host>;


//Wrappers to get nices names and colors in Nsight System
template <class Layout, class ExecSpace>
std::string message_generator(const std::string& kernel_name) {
  std::string message;

  if (std::is_same<Layout, LayoutLeft>::value && std::is_same<ExecSpace, Device>::value) {
    message = "LL Device";
  }
  else if (std::is_same<Layout, LayoutRight>::value && std::is_same<ExecSpace, Device>::value) {
    message = "LR Device";
  }
  else if (std::is_same<Layout, LayoutLeft>::value && std::is_same<ExecSpace, Host>::value) {
    message = "LL Host";
  }
  else if (std::is_same<Layout, LayoutRight>::value && std::is_same<ExecSpace, Host>::value) {
    message = "LR Host";
  }
  else {
    message = "ERROR";
  }

  // Prepend the kernel name followed by a space
  return kernel_name + " " + message;
}

template <class ExecSpace>
std::string color_generator() {

  if (std::is_same<ExecSpace, Device>::value) {
    return "green";
  }
  else if (std::is_same<ExecSpace, Host>::value) {
    return "cyan";
  }
  else {
    return "ERROR";
  }
}

//kernel that we are timing
//Inputs are the layout, the execution space and the number of launches
template <class Layout, class ExecSpace>
void blurrKernel(Kokkos::View<double**, Layout, ExecSpace> &view, const int nlaunch=1){

  const int N0=view.extent(0);
  const int N1=view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({1, 1}, {N0-1, N1-1});

  auto message =message_generator<Layout, ExecSpace>(__func__);
  auto color =color_generator<ExecSpace>();

  Kokkos::fence();
  mynvtxRangePush(message, color);

  for (int i=0; i<nlaunch; i++){
  Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j)
      {
        view(i,j)=(1.0/5)*(view(i-1,j)+view(i,j)+view(i+1,j)+view(i,j-1)+view(i,j+1));
    });
  Kokkos::fence();
  }

  Kokkos::fence();
  mynvtxRangePop();
}

//kernel that we are timing
//Inputs are the layout, the execution space
template <class Layout, class ExecSpace>
void InitKernel(Kokkos::View<double**, Layout, ExecSpace> &view){

  const int N0=view.extent(0);
  const int N1=view.extent(1);

  Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> policy({0, 0}, {N0, N1});

  auto message =message_generator<Layout, ExecSpace>(__func__);
  auto color =color_generator<ExecSpace>();

  Kokkos::fence();
  mynvtxRangePush(message, color);

  Kokkos::parallel_for(message, policy, KOKKOS_LAMBDA(const int i, const int j)
      {
        view(i,j)=1.0;
    });

  Kokkos::fence();
  mynvtxRangePop();
}



int main(int argc, char* argv[]) {

    // Initialize Kokkos runtime
    mynvtxRangePush("Kokkos::initialize", "blue");
    Kokkos::initialize(argc, argv);
    mynvtxRangePop();

    //Main scope
    mynvtxRangePush("Main scope", "purple");
    {

      //Size of the view
      const int N0=10000;
      const int N1=100000;

      //Allocate the views
      mynvtxRangePush("alloc LL Device", "green");
      auto device_view_LL = ViewLLDevice("device_view_LL", N0, N1);
      mynvtxRangePop();

      mynvtxRangePush("alloc LR Device", "green");
      auto device_view_LR = ViewLRDevice("device_view_LR", N0, N1);
      mynvtxRangePop();

      mynvtxRangePush("alloc LL Host", "cyan");
      auto host_view_LL = ViewLLHost("host_view_LL", N0, N1);
      mynvtxRangePop();

      mynvtxRangePush("alloc LR Host", "cyan");
      auto host_view_LR = ViewLRHost("host_view_LL", N0, N1);
      mynvtxRangePop();

      //Launch init kernels
      InitKernel<LayoutLeft, Device>(device_view_LL);
      InitKernel<LayoutRight, Device>(device_view_LR);
      InitKernel<LayoutLeft, Host>(host_view_LL);
      InitKernel<LayoutRight, Host>(host_view_LR);

      //Launch blurr kernels
      const int nlaunch_gpu=10;
      blurrKernel<LayoutLeft, Device>(device_view_LL, nlaunch_gpu);
      blurrKernel<LayoutRight, Device>(device_view_LR, nlaunch_gpu);

      const int nlaunch_cpu=3;
      blurrKernel<LayoutLeft, Host>(host_view_LL, nlaunch_cpu);
      blurrKernel<LayoutRight, Host>(host_view_LR, nlaunch_cpu);

    }
    mynvtxRangePop();

    // Finalize Kokkos runtime
    mynvtxRangePush("Kokkos::finalize", "red");
    Kokkos::finalize();
    mynvtxRangePop();

    return 0;
}       
