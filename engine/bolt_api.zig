//! libbolt's C ABI as the Bolt backend sees it: the symbols, resolved once, and where the library is found.
const std = @import("std");
const builtin = @import("builtin");

/// The C ABI version of libbolt this backend speaks (`bolt_abi`).
pub const abi_version: c_int = 3;

const windows = builtin.os.tag == .windows;
/// libbolt.so on Linux, bolt.dll on Windows.
pub const file_name = if (windows) "bolt.dll" else "libbolt.so";

/// dlopen and its kin, or the Win32 loader.
const os = if (windows) struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;
    extern "kernel32" fn GetModuleFileNameA(module: ?*anyopaque, name: [*]u8, size: u32) callconv(.winapi) u32;
    fn open(path: [*:0]const u8) ?*anyopaque {
        return LoadLibraryA(path);
    }
    fn sym(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque {
        return GetProcAddress(handle orelse return null, name);
    }
    fn err() ?[*:0]const u8 {
        return null;
    }
    fn exe(buf: []u8) ?[]const u8 {
        const n = GetModuleFileNameA(null, buf.ptr, @intCast(buf.len));
        return if (n == 0 or n >= buf.len) null else buf[0..n];
    }
} else struct {
    extern "c" fn dlopen(path: [*:0]const u8, mode: c_int) ?*anyopaque;
    extern "c" fn dlsym(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque;
    // libc's dlerror, named apart from this file's dlerror() wrapper
    const c_dlerror = @extern(*const fn () callconv(.c) ?[*:0]const u8, .{ .name = "dlerror" });
    extern "c" fn readlink(path: [*:0]const u8, buf: [*]u8, len: usize) isize;
    fn open(path: [*:0]const u8) ?*anyopaque {
        return dlopen(path, 2);
    }
    fn sym(handle: ?*anyopaque, name: [*:0]const u8) ?*anyopaque {
        return dlsym(handle, name);
    }
    fn err() ?[*:0]const u8 {
        return c_dlerror();
    }
    fn exe(buf: []u8) ?[]const u8 {
        const n = readlink("/proc/self/exe", buf.ptr, buf.len - 1);
        return if (n > 0) buf[0..@intCast(n)] else null;
    }
};

/// Why the library did not open, when the loader says (dlerror; Windows gives no text here).
pub fn dlerror() ?[*:0]const u8 {
    return os.err();
}

/// A run of a prompt piece whose embeddings are rows on the device (bolt_prefill_seq_rows).
pub const RowRun = extern struct { pos: c_int, count: c_int, rows: u64 };

/// libbolt's C ABI, resolved once.
pub const Api = struct {
    open: *const fn ([*:0]const u8, c_int) callconv(.c) ?*anyopaque,
    free: *const fn (?*anyopaque) callconv(.c) void,
    seqs: *const fn (?*anyopaque) callconv(.c) c_int,
    seq_alloc: *const fn (?*anyopaque) callconv(.c) c_int,
    seq_release: *const fn (?*anyopaque, c_int) callconv(.c) void,
    seq_bytes: *const fn (?*anyopaque) callconv(.c) u64,
    prompt_end: *const fn (?*anyopaque, c_int, c_int) callconv(.c) void,
    prefill: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int,
    forward: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, c_int) callconv(.c) c_int,
    forward_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int,
    candidates: *const fn (?*anyopaque, c_int, c_int, [*]f32, [*]c_int) callconv(.c) void,
    keep: *const fn (?*anyopaque, c_int, c_int) callconv(.c) void,
    keep_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int) callconv(.c) void,
    propose: *const fn (?*anyopaque, c_int, c_int, [*]c_int) callconv(.c) c_int,
    propose_multi: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]c_int, [*]c_int) callconv(.c) void,
    has_draft: *const fn (?*anyopaque) callconv(.c) c_int,
    abi: *const fn () callconv(.c) c_int,
    mask_rows: *const fn (?*anyopaque, c_int, [*]const c_int, [*]const u32, c_int, c_int) callconv(.c) c_int,
    prefill_rows: *const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, [*]const RowRun, c_int) callconv(.c) c_int,
    cost: *const fn (?*anyopaque, c_int) callconv(.c) f64,
    draft_multi_cost: *const fn (?*anyopaque) callconv(.c) f64,
    shared_costs: *const fn (?*anyopaque, [*]c_int, [*]f64, c_int) callconv(.c) c_int,
    snap_bytes: *const fn (?*anyopaque, c_int) callconv(.c) u64,
    snap_save: *const fn (?*anyopaque, c_int, c_int) callconv(.c) ?*anyopaque,
    snap_restore: *const fn (?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int,
    snap_drop: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) void,

    pub fn load(handle: ?*anyopaque) !Api {
        return .{
            .open = try sym(*const fn ([*:0]const u8, c_int) callconv(.c) ?*anyopaque, handle, "bolt_open"),
            .free = try sym(*const fn (?*anyopaque) callconv(.c) void, handle, "bolt_free"),
            .seqs = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_seqs"),
            .seq_alloc = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_seq_alloc"),
            .seq_release = try sym(*const fn (?*anyopaque, c_int) callconv(.c) void, handle, "bolt_seq_release"),
            .seq_bytes = try sym(*const fn (?*anyopaque) callconv(.c) u64, handle, "bolt_seq_bytes"),
            .prompt_end = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) void, handle, "bolt_prompt_end"),
            .prefill = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int) callconv(.c) c_int, handle, "bolt_prefill_seq"),
            .forward = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, c_int) callconv(.c) c_int, handle, "bolt_forward_seq"),
            .forward_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]const c_int, c_int) callconv(.c) c_int, handle, "bolt_forward_multi"),
            .candidates = try sym(*const fn (?*anyopaque, c_int, c_int, [*]f32, [*]c_int) callconv(.c) void, handle, "bolt_candidates"),
            .keep = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) void, handle, "bolt_keep_seq"),
            .keep_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int) callconv(.c) void, handle, "bolt_keep_multi"),
            .propose = try sym(*const fn (?*anyopaque, c_int, c_int, [*]c_int) callconv(.c) c_int, handle, "bolt_propose_seq"),
            .propose_multi = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const c_int, [*]c_int, [*]c_int) callconv(.c) void, handle, "bolt_propose_multi"),
            .has_draft = try sym(*const fn (?*anyopaque) callconv(.c) c_int, handle, "bolt_has_draft"),
            .abi = try sym(*const fn () callconv(.c) c_int, handle, "bolt_abi"),
            .mask_rows = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, [*]const u32, c_int, c_int) callconv(.c) c_int, handle, "bolt_mask_rows"),
            .prefill_rows = try sym(*const fn (?*anyopaque, c_int, [*]const c_int, c_int, c_int, [*]const RowRun, c_int) callconv(.c) c_int, handle, "bolt_prefill_seq_rows"),
            .cost = try sym(*const fn (?*anyopaque, c_int) callconv(.c) f64, handle, "bolt_cost"),
            .draft_multi_cost = try sym(*const fn (?*anyopaque) callconv(.c) f64, handle, "bolt_draft_multi_cost"),
            .shared_costs = try sym(*const fn (?*anyopaque, [*]c_int, [*]f64, c_int) callconv(.c) c_int, handle, "bolt_shared_costs"),
            .snap_bytes = try sym(*const fn (?*anyopaque, c_int) callconv(.c) u64, handle, "bolt_snap_bytes"),
            .snap_save = try sym(*const fn (?*anyopaque, c_int, c_int) callconv(.c) ?*anyopaque, handle, "bolt_snap_save_seq"),
            .snap_restore = try sym(*const fn (?*anyopaque, c_int, ?*anyopaque) callconv(.c) c_int, handle, "bolt_snap_restore_seq"),
            .snap_drop = try sym(*const fn (?*anyopaque, ?*anyopaque) callconv(.c) void, handle, "bolt_snap_drop"),
        };
    }

    fn sym(comptime T: type, handle: ?*anyopaque, name: [*:0]const u8) !T {
        const p = os.sym(handle, name) orelse {
            std.log.err("bolt: {s} has no {s}", .{ file_name, name });
            return error.BoltMissing;
        };
        return @ptrCast(@alignCast(p));
    }
};

/// libbolt: TENSORFOLD_BOLT_LIB, else beside the server (native/lib), else the loader's path.
pub fn openLibrary() ?*anyopaque {
    if (@import("sys").getenv("TENSORFOLD_BOLT_LIB")) |p| return os.open(p.ptr);
    var exe: [std.fs.max_path_bytes]u8 = undefined;
    if (os.exe(&exe)) |self| {
        const dir = std.fs.path.dirname(self) orelse "";
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        if (std.fmt.bufPrintSentinel(&buf, "{s}/../lib/" ++ file_name, .{dir}, 0)) |path| {
            if (os.open(path)) |h| return h;
        } else |_| {}
    }
    return os.open(file_name);
}
