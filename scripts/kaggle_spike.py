"""Phase 0 Kaggle spike: can we build and run LibTorch C++ against Kaggle's pip torch on a T4?

Paste into a Kaggle notebook cell (GPU T4, Internet on) and run. It is self-contained:
it records the environment, writes a tiny CMake project, builds it, and runs it on CUDA.
Copy the full output back into IMPLEMENTATION.md section 2.
"""
import os
import shutil
import subprocess
import sys
import textwrap

import torch


def sh(cmd, check=True, **kw):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out[-6000:], flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {cmd}")
    return r


print("=== environment ===")
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| cudnn:", torch.backends.cudnn.version())
print("cxx11 abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("cmake prefix:", torch.utils.cmake_prefix_path)
print("gpu:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
print("cpu count:", os.cpu_count())
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv", check=False)
sh("nvcc --version || ls /usr/local/cuda*/bin/nvcc", check=False)
sh("gcc --version | head -1; cmake --version | head -1; ninja --version", check=False)
sh("free -g | head -2; df -h /kaggle/working | tail -1", check=False)
sh("cargo --version", check=False)

if shutil.which("cmake") is None or shutil.which("ninja") is None:
    sh(f"{sys.executable} -m pip install -q cmake ninja")

print("\n=== build hello_cuda ===")
root = "/kaggle/working/spike"
os.makedirs(root, exist_ok=True)
with open(f"{root}/CMakeLists.txt", "w") as f:
    f.write(textwrap.dedent("""\
        cmake_minimum_required(VERSION 3.24)
        project(spike LANGUAGES CXX)
        set(CMAKE_CXX_STANDARD 17)
        find_package(Torch REQUIRED)
        add_executable(hello_cuda main.cpp)
        target_link_libraries(hello_cuda PRIVATE ${TORCH_LIBRARIES})
    """))
with open(f"{root}/main.cpp", "w") as f:
    f.write(textwrap.dedent("""\
        #include <torch/torch.h>
        #include <torch/version.h>
        #include <iostream>
        int main() {
          std::cout << "LibTorch " << TORCH_VERSION << " cuda=" << torch::cuda::is_available() << "\\n";
          torch::InferenceMode g;
          auto x = torch::randn({1024, 1024}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
          auto y = torch::mm(x, x);
          torch::cuda::synchronize();
          std::cout << y.device() << " " << y.dtype() << " sum=" << y.sum().item<float>() << "\\n";
          auto q = torch::randn({8, 12, 128, 64}, torch::device(torch::kCUDA).dtype(torch::kFloat16));
          auto a = torch::scaled_dot_product_attention(q, q, q, {}, 0.0, true);
          std::cout << "sdpa ok " << a.sizes() << "\\n";
          return 0;
        }
    """))

cuda_args = ""
nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
if nvcc:
    cuda_args = f"-DCMAKE_CUDA_COMPILER={nvcc} -DCUDAToolkit_ROOT={os.path.dirname(os.path.dirname(nvcc))}"
env = dict(os.environ, TORCH_CUDA_ARCH_LIST="7.5")
r = sh(
    f"cmake -S {root} -B {root}/build -G Ninja -DCMAKE_BUILD_TYPE=Release "
    f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path} {cuda_args}",
    check=False, env=env,
)
if r.returncode == 0:
    sh(f"cmake --build {root}/build -j {os.cpu_count()}", env=env)
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    run_env = dict(os.environ, LD_LIBRARY_PATH=f"{torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}")
    sh(f"{root}/build/hello_cuda", env=run_env)
    print("\nSPIKE PASSED")
else:
    print("\nSPIKE FAILED at configure step - copy the output above")
