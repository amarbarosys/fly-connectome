"""The engine GUI's Brain panel (live.mojo -DGUI=1): the selected fly's brain drawn live (kinestride brain_view.mojo,
from the worker's FLY_TAP), what its named groups do right now (brain_layout.py's output groups: the DNs that walk,
turn, escape, back up; every input the keys and loom fire), and a log of when each group switched on.

Files: BRAIN_LAYOUT dir (xyz.bin, class.bin, groups.bin, groups.txt), FLY_TAP path (header seq, column, tick, N; then
N activity bytes) and FLY_TAP.sel (int32: the column the worker taps = the selected fly). Read into one buffer
allocated at start; the view is redrawn only when the tap's seq or the camera changed.
"""
from std.ffi import external_call
from std.sys import stderr
from max.gpu.host import DeviceBuffer, DeviceContext
from brain_view import BH, BW, BrainView
import imkit.ui as uim
from imkit.window import Input

comptime X0 = 20
comptime Y0 = 108
comptime X1 = 400
comptime IX = 30  # image at (IX, IY), BW x BH
comptime IY = 144


struct BrainPanel(Movable):
    var bv: BrainView
    var tap: String
    var buf: List[UInt8]  # 16-byte header + N activity bytes
    var names: List[String]
    var effects: List[String]
    var outputs: List[Bool]
    var start: List[Int]  # group g's ids: ids[start[g]:start[g + 1]]
    var ids: List[Int]
    var on: List[Bool]
    var events: List[String]
    var sel: Int
    var seq: Int32
    var yaw: Float32
    var pitch: Float32
    var drag: Bool

    def __init__(out self, ctx: DeviceContext, layout: String, tap: String) raises:
        self.bv = BrainView(ctx, layout)
        self.tap = tap.copy()
        self.buf = List[UInt8](length=16 + self.bv.n, fill=0)
        self.names = List[String]()
        self.effects = List[String]()
        self.outputs = List[Bool]()
        for line in open(layout + "groups.txt", "r").read().split("\n"):  # not splitlines(): Mojo 1.1's also splits at tabs
            if not line:
                continue
            var f = line.split("\t")
            self.names.append(String(f[0]))
            self.effects.append(String(f[1]))
            self.outputs.append(String(f[2]) == "output")
        var gb = open(layout + "groups.bin", "r").read_bytes()
        var g = gb.unsafe_ptr().unsafe_bitcast[Int32]()
        self.start = List[Int]()
        self.ids = List[Int]()
        var o = 1
        for _ in range(Int(g[0])):
            self.start.append(len(self.ids))
            for k in range(Int(g[o])):
                self.ids.append(Int(g[o + 1 + k]))
            o += 1 + Int(g[o])
        self.start.append(len(self.ids))
        self.on = List[Bool](length=len(self.names), fill=False)
        self.events = List[String]()
        self.sel, self.seq, self.yaw, self.pitch, self.drag = -1, -1, 0.35, 0.25, False
        self.select(0)

    def select(mut self, f: Int) raises:
        """Tap fly f's column from the next tap on (the worker reads FLY_TAP.sel)."""
        if f == self.sel:
            return
        self.sel = f
        var v = List[Int32](length=1, fill=Int32(f))
        with open(self.tap + ".sel", "w") as o:
            o.write_bytes(Span(unsafe_ptr=v.unsafe_ptr().unsafe_bitcast[UInt8](), length=4))
        for k in range(len(self.on)):
            self.on[k] = False

    def read(mut self) -> Bool:
        """The newest tap into buf; True when it is a new one for the selected fly."""
        var path = self.tap.copy()
        var fd = external_call["open64", Int32](path.as_c_string_slice().unsafe_ptr(), Int32(0))  # open64: the stdlib declares open itself
        if fd < 0:
            return False
        var n = external_call["read", Int](Int(fd), self.buf.unsafe_ptr().unsafe_bitcast[NoneType](), len(self.buf))  # live.read_all's types
        _ = external_call["close", Int32](fd)
        var h = self.buf.unsafe_ptr().unsafe_bitcast[Int32]()
        if n != len(self.buf) or Int(h[1]) != self.sel or h[0] == self.seq:
            return False
        self.seq = h[0]
        return True

    def frame(mut self, ctx: DeviceContext, mut u: uim.Ui, inp: Input, src: DeviceBuffer[DType.uint32], t: Float64) raises:
        var fresh = self.read()
        u.panel(X0, Y0, X1, Float32(uim.UH - 20), "Brain   Fly " + String(self.sel) + "   (drag to turn)")
        var h = u.hovered(IX, IY, Float32(IX + BW), Float32(IY + BH))
        if inp.pressed & 1:
            self.drag = h
        if not (inp.down & 1):
            self.drag = False
        var moved = self.drag and (inp.dx != 0 or inp.dy != 0)
        if moved:
            self.yaw += inp.dx * 0.01
            self.pitch = min(max(self.pitch + inp.dy * 0.01, Float32(-1.5)), Float32(1.5))
        if fresh or moved:
            self.bv.draw(ctx, Int(self.buf.unsafe_ptr()) + 16 if fresh else 0, self.yaw, self.pitch, 150)
            ctx.enqueue_copy(src.create_sub_buffer[DType.uint32](0, BW * BH), self.bv.img)
        u.image(IX, IY, Float32(IX + BW), Float32(IY + BH), 0, BW, BH)
        u.x, u.y = Float32(IX), Float32(IY + BH + 10)
        var act = self.buf.unsafe_ptr() + 16
        for g in range(len(self.names)):
            var n = self.start[g + 1] - self.start[g]
            var lit = 0
            for k in range(self.start[g], self.start[g + 1]):
                if act[self.ids[k]] > 128:
                    lit += 1
            var now = lit >= max(1, n // 20)  # a few neurons of a big group, any of a small one
            if now and not self.on[g]:
                self.events.insert(0, "t " + uim.fmt(t) + " s   " + self.names[g] + "  ->  " + self.effects[g])
                print("brain fly", self.sel, self.events[0], file=stderr)  # checks read these (DIR/play.stderr)
                if len(self.events) > 4:
                    _ = self.events.pop()
            self.on[g] = now
            if not self.outputs[g]:
                continue
            var c = uim.ACCENT if now else uim.MUTED
            _ = u.text(u.x, u.y, self.names[g], 0, c)
            u.rect(u.x + 96, u.y + 5, u.x + 236, u.y + 11, 3, uim.SURFACE)
            u.rect(u.x + 96, u.y + 5, u.x + 96 + 140 * Float32(lit) / Float32(max(n, 1)), u.y + 11, 3, uim.BAD if now else uim.ACCENT)
            if now:
                _ = u.text(u.x + 246, u.y, self.effects[g], 0, uim.TEXT)
            u.y += 18
        u.y += 6
        for e in self.events:
            _ = u.text(u.x, u.y, e, 0, uim.TEXT)
            u.y += 17
