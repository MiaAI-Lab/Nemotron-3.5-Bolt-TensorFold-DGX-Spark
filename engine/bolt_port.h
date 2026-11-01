// The POSIX file calls libbolt reads checkpoints with: open, fstat, mmap, madvise, munmap, close.
// On Linux these are the system's own; on Windows they are mapped onto the CRT and Win32 file mappings.
#pragma once

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// 64-bit sizes: shards are up to 1.6 GB, past MSVC's 32-bit struct stat
#define stat _stat64
#define fstat _fstat64
#define open(path, flags) _open((path), (flags) | _O_BINARY)
#define close _close

#define PROT_READ 1
#define MAP_PRIVATE 2
#define MAP_FAILED ((void*)-1)
#define MADV_WILLNEED 3

// A read-only view of the whole file. The view keeps the file open after the descriptor and mapping handle close.
static inline void* mmap(void*, size_t len, int, int, int fd, long long) {
    HANDLE file = (HANDLE)_get_osfhandle(fd);
    if (file == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) return MAP_FAILED;
    void* p = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len);
    CloseHandle(mapping);
    return p ? p : MAP_FAILED;
}

static inline int munmap(void* p, size_t) { return UnmapViewOfFile(p) ? 0 : -1; }

// MADV_WILLNEED: ask Windows to read the range ahead, as Linux does
static inline int madvise(void* p, size_t len, int) {
    WIN32_MEMORY_RANGE_ENTRY range = {p, len};
    return PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0) ? 0 : -1;
}
#endif
