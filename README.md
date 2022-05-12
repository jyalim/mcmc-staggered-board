MCMC Staggered Board
====================

TODO

Profiling with NightSight Systems application
----------------------------------------------

Files are under `versions/gpu-v0.1nsight/`

1. Download from NVIDA Night Sight systems page (https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2022-2).
2. Install the nvtx and torch modules using the pip3 commands in your environment.<br />
   pip3 install nvtx <br />
   pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
3. Add the function nvtx.range_push, nvtx.range_pop between the python function for profiling. <br />
   ###### nvtx.range_push("name of this range/function") <br />
   ###### func() <br/>
   ###### nvtx.range_pop() <br/>
   For more references, view the NightSight systems training documentation  at the below link. <br />
   https://github.com/NVIDIA/nsight-training/blob/master/cuda/2021_gtc/x-ac-03-v1/task1/task/nsys/04_cpu_bottleneck.ipynb <br />
4. After updating with nvtx utilities function, start profiling with nsys cmds and the NightSight Systems application. <br />
   ###### nsys profile --trace cuda,osrt,nvtx  -o <output_file> --stats=true python likelihood_func1.py <iterations> <br />
   nsys cmds documentation - https://docs.nvidia.com/nsight-systems/UserGuide/index.html#example-single-command-lines

5. Open the output files(.qdrep) using the NightSight system UI and all the python profiling functions would be available under the NVTX section.
