{ pkgs, lib, ... }:

{
  env = {
    CUDA_PATH = "${pkgs.cudaPackages_12_1.cudatoolkit}";
    LD_LIBRARY_PATH = lib.makeLibraryPath [
      "/usr/lib/wsl/lib"
      "${pkgs.cudaPackages_12_1.cudatoolkit}"
      "${pkgs.cudaPackages_12_1.cudatoolkit.lib}"
      "${pkgs.cudaPackages_12_1.cuda_cudart}/lib"
      "${pkgs.cudaPackages_12_1.cuda_cupti}/lib"
      "${pkgs.cudaPackages_12_1.cuda_nvrtc}/lib"
      "${pkgs.linuxPackages.nvidia_x11}/lib"
      "${pkgs.ncurses5}/lib"
      "${pkgs.stdenv.cc.cc.lib}/lib"
      "${pkgs.zlib}/lib"
    ];
    EXTRA_LDFLAGS = "-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
    EXTRA_CCFLAGS = "-I/usr/include";
    CUDA_HOME = "${pkgs.cudaPackages_12_1.cudatoolkit}";
    NVIDIA_CUDA_HOME = "${pkgs.cudaPackages_12_1.cudatoolkit}";
    CUDA_TOOLKIT_ROOT_DIR = "${pkgs.cudaPackages_12_1.cudatoolkit}";
    
    CUDA_MODULE_LOADING = "LAZY";
    NVIDIA_DRIVER_CAPABILITIES = "compute,utility";
    NVIDIA_VISIBLE_DEVICES = "all";
    PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512";
  };

  packages = with pkgs; [
    cudaPackages_12_1.cudatoolkit
    cudaPackages_12_1.cuda_cudart
    cudaPackages_12_1.cuda_cupti
    cudaPackages_12_1.cuda_nvrtc
    linuxPackages.nvidia_x11
    libGL
    libGLU
    stdenv.cc
    zlib
  ];

  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ./requirements.txt;
  };

  dotenv.enable = true;
}
