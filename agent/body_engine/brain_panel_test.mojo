"""brain_panel.mojo without a window (GPU, under gpu-wait): the FlyWire layout, a fake tap with DNp09 lit for column 1,
fly 1 selected, one UI frame. Gates: the panel reads the tap, lights DNp09 -> walk forward (event), draws commands,
drops none; the brain view lands in the image sources.

  build: mojo build -I ~/Projects/mojo/kinestride/kinestride -I ~/Projects/mojo/mojo-imkit/src agent/body_engine/brain_panel_test.mojo
  run:   gpu-wait run --priority 90 --vram 1 --timeout 300 -- BINARY LAYOUT_DIR/ TAP_PATH
"""
from std.sys import argv
from max.gpu.host import DeviceContext
from brain_panel import BrainPanel
from brain_view import BH, BW
import imkit.ui as uim
from imkit.font import Font
from imkit.window import Input


def main() raises:
    var layout = String(argv()[1])
    var tap = String(argv()[2])
    var ctx = DeviceContext()
    var bp = BrainPanel(ctx, layout, tap)
    bp.select(1)
    var n = bp.bv.n
    var buf = List[UInt8](length=16 + n, fill=0)
    var h = buf.unsafe_ptr().unsafe_bitcast[Int32]()
    h[0], h[1], h[2], h[3] = 7, 1, 1000, Int32(n)
    for k in range(bp.start[0], bp.start[1]):  # group 0: DNp09
        buf[16 + bp.ids[k]] = 255
    with open(tap, "w") as o:
        o.write_bytes(Span(buf))
    var src = ctx.enqueue_create_buffer[DType.uint32](uim.ISRC)
    src.enqueue_fill(0)
    var u = uim.Ui(Font("/usr/share/fonts/TTF/Roboto-Medium.ttf", [13, 16, 22]))
    var inp = Input(600, 600, 0, 0, 0, 0, 0, 0, SIMD[DType.uint64, 2](0), False)
    u.begin(inp)
    bp.frame(ctx, u, inp, src, 1.0)
    u.finish()
    var img = List[UInt32](length=BW * BH, fill=0)
    ctx.enqueue_copy(img.unsafe_ptr(), src.create_sub_buffer[DType.uint32](0, BW * BH))
    ctx.synchronize()
    var drawn = 0
    for p in img:
        if p != 0:
            drawn += 1
    print("events", len(bp.events), bp.events[0] if len(bp.events) else "", "; commands", u.n, "dropped", u.dropped, "; image pixels set", drawn)
    if len(bp.events) == 0 or "DNp09  ->  walk forward" not in bp.events[len(bp.events) - 1] and "DNp09  ->  walk forward" not in bp.events[0] or u.dropped or drawn != BW * BH:
        raise Error("FAIL brain_panel_test")
    print("PASS brain_panel_test")
