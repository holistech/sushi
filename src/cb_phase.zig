//! holistech fork: the command-buffer MB limit per phase on the M1 Ultra.
//!
//! MLX commits a command buffer once its bound buffers pass a size limit, read once when the Metal
//! device is created. On the M1 Ultra a long prefill wants a small limit, decode a large one
//! (sushi-schalter.py, 32k document, no MTP, 2026-10-17: 1000 MB prefill 629 / decode 45.5 tok/s,
//! 250 MB 650 / 43.8, 100 MB 650 / 42.0). The fork's MLX exports a setter
//! (patches/mlx-cb-phase-setter.patch), found by dlsym so a stock libmlx keeps the fixed value.
//! Switching per phase, A B B A over the API: 32k prefill +2.7 % (no MTP) and +3.6 % (MTP), decode
//! and greedy text unchanged. Only the inference thread calls `enter`.
const std = @import("std");

pub const Phase = enum { none, prefill, decode };

const Setter = *const fn (c_int) callconv(.c) c_int;
var setter: ?Setter = null;
var mb_prefill: c_int = 0;
var mb_decode: c_int = 0;
var current: Phase = .none;

/// Arms the switch. Call before any MLX work; it only resolves the symbol.
pub fn enable(prefill_mb: c_int, decode_mb: c_int) void {
    const rtld_default: ?*anyopaque = @ptrFromInt(@as(usize, @bitCast(@as(isize, -2))));
    const sym = std.c.dlsym(rtld_default, "mlx_holistech_set_max_mb_per_buffer") orelse return;
    setter = @ptrCast(@alignCast(sym));
    mb_prefill = prefill_mb;
    mb_decode = decode_mb;
}

/// Switches to `p` and returns the phase it replaced (for `restore`). A no-op when not armed.
pub fn enter(p: Phase) Phase {
    const f = setter orelse return current;
    const prev = current;
    if (p != prev and p != .none) {
        _ = f(if (p == .prefill) mb_prefill else mb_decode);
        current = p;
    }
    return prev;
}

/// Returns to the phase `enter` replaced: a decode tick inside a prefill hands the limit back.
pub fn restore(p: Phase) void {
    _ = enter(p);
}
