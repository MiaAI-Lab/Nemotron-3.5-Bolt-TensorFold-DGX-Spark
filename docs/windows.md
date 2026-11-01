# Windows on ARM (NVIDIA RTX Spark N1X)

The same build without Docker: TensorFold at `TF_REF` with `patches/`, `engine/` compiled into `bolt.dll` beside
`tensorfold-native.exe`. Checked on one machine: an RTX Spark N1X laptop (sm_121, 64 GB, of which 40 GB were
reserved for the GPU), Windows 11 Pro ARM64, NVIDIA driver 621.01. TensorFold's own Windows support is
[docs/WINDOWS.md](https://github.com/ashhart/TensorFold/blob/main/docs/WINDOWS.md) in its tree.

## What it needs

- Visual Studio Build Tools 2022 with the C++ workload and the ARM64 tools (`cl.exe`, nvcc's host compiler).
- CUDA 13.4's nvcc for `windows-arm64`. NVIDIA's redistributable zips are enough: `cuda_nvcc`, `cuda_crt`,
  `cuda_cudart`, `cccl` and `libnvvm`, unpacked into one folder (`CUDA_PATH`).
- Zig 0.17.0 (`aarch64-windows`).
- Git with `core.autocrlf false` for this repository: `patches/` does not apply to TensorFold's LF files from a CRLF
  checkout.
- The Visual C++ runtime (`MSVCP140.dll`, `VCRUNTIME140.dll`), which `bolt.dll` links against.

## Build

In PowerShell, from this repository's root:

```powershell
# the ARM64 compiler's environment, from vcvarsall.bat into this PowerShell
cmd /c "`"<Build Tools>\VC\Auxiliary\Build\vcvarsall.bat`" arm64 >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] } }
git init tensorfold; cd tensorfold
git config core.autocrlf false
git remote add origin https://github.com/ashhart/TensorFold.git
git fetch --depth 1 origin eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab; git checkout --detach FETCH_HEAD
git apply ..\patches\0002-register-bolt.patch
copy ..\engine\bolt_native.zig, ..\engine\bolt_api.zig zig\src\families\nemotron\
zig build native bolt install -Dsm=121 -Dstrip=true "-Dnvcc=$env:CUDA_PATH\bin\nvcc.exe" "-Dbolt-src=$((Resolve-Path ..\engine).Path)"
```

The result is `zig-out\native\bin\tensorfold-native.exe` and `zig-out\native\lib\bolt.dll`, where the backend looks
for it (`TENSORFOLD_BOLT_LIB` names another path).

## Run

The checkpoint and the drafter as `scripts/prepare.sh` pins them, in a Hugging Face cache (`HF_HOME`). Anonymous
downloads are rate-limited to 3,000 lookups per 5 minutes, and `tensorfold-native pull`, which fetches a file in
many ranged pieces, ran into that limit on the 23 GiB checkpoint; one request a file (curl, or `hf download`)
does not.

```powershell
$env:BOLT_DRAFT = "<HF_HOME>\hub\models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr\snapshots\4c6a5613b6d887194a3a5493e3a18a5b3930bb86"
$env:BOLT_KV = 'bf16'; $env:BOLT_KV_PIN = '0'; $env:BOLT_SSM_LAZY = '1'; $env:BOLT_DRAFT_Q8 = '1'; $env:BOLT_NV4_FP4 = '1'
zig-out\native\bin\tensorfold-native.exe serve "<HF_HOME>\hub\models--nvidia--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB\snapshots\a326822af111290f2af5ac59ef4b5040347dabc4" `
    --name Nemotron-3.5-Bolt --host 127.0.0.1 --port 8080 --parallel 4 --context 131072 --prompt-cache-gib 4
```

These are `scripts/config.sh`'s defaults with a smaller `PARALLEL`, `CONTEXT` and `PROMPT_CACHE_GIB`. `start.sh` and its
`MemAvailable` checks are not used here; on this machine the GPU has its own reserved pool. Loading took 66 seconds.
With these settings the GPU held 33.2 GB, the display's share included.

## What is different on Windows

| File | Change |
| --- | --- |
| `engine/bolt_port.h` | `open`, `fstat`, `close`, `mmap`, `munmap` and `madvise` over the CRT and Win32 file mappings; the POSIX headers elsewhere |
| `engine/bolt_host.cu`, `engine/bolt_draft.cu` | include `bolt_port.h` instead of `fcntl.h`, `sys/mman.h`, `sys/stat.h`, `unistd.h` |
| `engine/bolt.def` | the exports of `bolt.dll`: the `extern "C"` functions of `bolt_host.cu` |
| `engine/bolt_api.zig` | `LoadLibraryA` / `GetProcAddress` / `GetModuleFileNameA` in place of `dlopen` / `dlsym` / `/proc/self/exe`; `bolt.dll` in `native\lib` |
| `engine/bolt_native.zig` | the clock through TensorFold's `sys.c.clock_gettime`, `setenv` through `sys.env.setenv`, `BOLT_STATS` through `sys.getenv`; `BOLT_SEQS` and `BOLT_DRAFT_OFF` are set before the library loads |
| `patches/0002` (`zig/build/bolt.zig`) | on Windows: `bolt.dll`, `-std=c++17` (gcc's default; `std::string::data()` is const before it), `-Xcompiler=/MD,/Zc:preprocessor,/EHsc` (the shared UCRT, so the server's `setenv` reaches the library; MSVC's conforming preprocessor for CUDA 13's headers), `-D_USE_MATH_DEFINES`, `-Xlinker=/DEF:bolt.def` |

On Linux the Zig files call what they called before, and `bolt.zig` passes the same nvcc arguments. The Linux build
has not been rebuilt with these changes.

## Measured

With the settings above, from a PowerShell client against the running server: thinking off, 512-token replies, one
run each (not medians). The laptop ran on battery in Windows' Balanced power plan.

| | TensorFold v1.1.8 alone | this recipe |
| --- | --- | --- |
| one stream, prose greedy / code greedy / prose sampled | 39.3 / 36.4 / 36.6 tok/s | 59.9 / 100.1 / 47.6 tok/s |
| four streams at once (greedy) | 91.7 tok/s | 117.6 tok/s (41.9 a stream) |
| a 24,267-token prompt | 1,583 tok/s | 1,508 tok/s |

Greedy replies with drafts equal `"draft": false` replies token for token (with the lazy-state fix of #<fix PR>).

Not built here: the video decoders (`libtfvideo`, FFmpeg), `libturbojpeg` and `libtfgrammar`, so neither pictures
nor structured output; plain chat is what was checked.
