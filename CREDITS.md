# Credits

- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors). This recipe pins
  `eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab` (v1.1.6): its Zig server, RADIO image tower, video decoder (`libtfvideo`)
  and [Nemotron recipes](https://github.com/ashhart/TensorFold/blob/eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab/docs/recipes/nemotron-3.5-super-vl.md).
- [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) by
  NVIDIA, revision `a326822af111290f2af5ac59ef4b5040347dabc4`. The weights are under the OpenMDW License Agreement,
  version 1.1, and are downloaded at setup. They are not part of this repository.
- The companion drafter
  [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)
  by NVIDIA, revision `4c6a5613b6d887194a3a5493e3a18a5b3930bb86`. It is downloaded at setup and runs in `engine/bolt_draft.cu`.
- The vision tower in the checkpoint is NVIDIA's C-RADIO v4-H; TensorFold's `families/radio` runs it, and the video
  sampling and labels follow the checkpoint's published processor (`processing_nemotron_h_omni.py`).
- [ggml](https://github.com/ggml-org/ggml) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the IQ2_XXS and
  Q8_0 formats the checkpoint is packed in, and the IQ2_XXS tables in `engine/iq2_tables.inc`.
- [FFmpeg](https://ffmpeg.org) 9.0.2 (decoders only, built from source) inside `libtfvideo`, and
  [libjpeg-turbo](https://libjpeg-turbo.org) 3.1.4.1, which the image builds for TensorFold's JPEG decoder.
- Christian Schröder ([cwschroeder](https://github.com/cwschroeder)) reviewed the video change for TensorFold
  ([#940](https://github.com/ashhart/TensorFold/pull/940)): the patch that keeps user text spelling the video token
  ordinary (and its tests) is theirs, carried in `patches/0003` until TensorFold 1.1.0 took it in (#940), and their reproducer found the rows-buffer sizing bug fixed there.
- [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8 (Apache-2.0), with [picojson](https://github.com/kazuho/picojson)
  (BSD-2-Clause) and [dlpack](https://github.com/dmlc/dlpack) (Apache-2.0): TensorFold's `libtfgrammar`, which the image
  builds so structured output and forced tool calls work. xgrammar's NOTICE ships in the image.
- [Zig](https://ziglang.org) 0.17.0 by the Zig Software Foundation and the Zig contributors.
- NVIDIA's [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) `26.07-py3`, the image
  base. Its license is in [`NOTICE`](NOTICE).
- The script layout (start, stop, prepare, config, banner, checkpoint check, image labels) follows MiaAI-Lab's
  [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold).
