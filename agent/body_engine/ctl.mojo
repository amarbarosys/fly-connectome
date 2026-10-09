"""RuntimeControl client in the engine (agent/runtime_control.py's abstract unix socket, the panel's own path), so the
GUI's buttons act exactly as /panel does: the same actions, acknowledged by EngineControl at a control-step boundary.
Never blocks the frame: send() connects and writes, pump() collects whatever replies have arrived (a step is
acknowledged only after its steps ran, which needs this engine to keep stepping meanwhile). The session id comes
from DIR/control.sid (brain_watch writes it), read lazily.

    ctl DIR/ ACTION [EDIT_JSON]     one request and its reply on stdout (tests/test_engine_control.py drives it)
"""
from std.ffi import external_call
from std.sys import argv
from std.time import perf_counter_ns
from window import readable


struct Ctl(Movable):
    var dir: String
    var sid: String
    var pending: List[Int]
    var seq: Int
    var reply: String  # the newest reply: EngineControl's snapshot

    def __init__(out self, dir: String):
        self.dir = dir.copy()
        self.sid = String()
        self.pending = List[Int]()
        self.seq = 0
        self.reply = String()

    def send(mut self, action: String, edit: String = "", steps: Int = 1):
        """Queue one request; never waits: a socket whose listener is busy (backlog full, runner not serving yet) is
        dropped, a status is not asked while one is pending."""
        if action == "status" and len(self.pending):
            return
        if not self.sid:
            try:
                self.sid = String(open(self.dir + "control.sid", "r").read().strip())
            except:
                return
        var fd = Int(external_call["socket", Int32](Int32(1), Int32(5 | 0x800), Int32(0)))  # AF_UNIX, SOCK_SEQPACKET | SOCK_NONBLOCK
        var name = String("fly-control-") + self.sid
        var addr = List[UInt8](length=110, fill=0)  # sockaddr_un: family 1, then '\0' + name (abstract)
        addr[0] = 1
        for k, ch in enumerate(name.as_bytes()):
            addr[3 + k] = ch
        if fd < 0 or external_call["connect", Int32](Int32(fd), addr.unsafe_ptr(), UInt32(3 + name.byte_length())) != 0:
            _ = external_call["close", Int32](Int32(fd))
            return
        self.seq += 1
        var h = hex(perf_counter_ns() * 1000 + self.seq)
        var rid = String("0") * (34 - h.byte_length()) + String(h[byte=2:])
        var req = String('{"session_id": "') + self.sid + '", "request_id": "' + rid + '", "action": "' + action + '", "steps": ' + String(steps) + \
            (String(', "edit": ') + edit if edit else String()) + "}"
        _ = external_call["send", Int](Int32(fd), req.unsafe_ptr(), req.byte_length(), Int32(0))
        self.pending.append(fd)

    def pump(mut self):
        var buf = List[UInt8](length=65536, fill=0)
        var keep = List[Int]()
        for fd in self.pending:
            if readable(fd, 0):
                var n = Int(external_call["recv", Int](Int32(fd), buf.unsafe_ptr(), 65536, Int32(0)))
                if n > 0:
                    self.reply = String(StringSlice(unsafe_from_utf8=Span(buf)[:n]))
                _ = external_call["close", Int32](Int32(fd))
            else:
                keep.append(fd)
        self.pending = keep^

    def num(self, key: String, start: Int = 0) -> Tuple[Float64, Int]:
        """The number after '"key": ' at or after byte start, and the byte after it (-1: absent)."""
        var k = self.reply.find(String('"') + key + '": ', start)
        if k < 0:
            return (Float64(0), -1)
        var a = k + key.byte_length() + 4
        var b = a
        var s = self.reply.as_bytes()
        while b < len(s) and (Int(s[b]) == ord("-") or Int(s[b]) == ord(".") or Int(s[b]) == ord("e") or (Int(s[b]) >= ord("0") and Int(s[b]) <= ord("9"))):
            b += 1
        try:
            return (atof(String(self.reply[byte=a:b])), b)
        except:
            return (Float64(0), -1)

    def nums(self, key: String) -> List[Float64]:
        """Every number inside the bracket after '"key": [' (a flat read of nested lists: drive [[L, R], ...])."""
        var out = List[Float64]()
        var k = self.reply.find(String('"') + key + '": [')
        if k < 0:
            return out^
        var s = self.reply.as_bytes()
        var i = k + key.byte_length() + 5
        var depth = 1
        while i < len(s) and depth > 0:
            var c = Int(s[i])
            if c == ord("["):
                depth += 1
            elif c == ord("]"):
                depth -= 1
            elif c == ord("-") or (c >= ord("0") and c <= ord("9")):
                var j = i
                while j < len(s) and (Int(s[j]) == ord("-") or Int(s[j]) == ord(".") or Int(s[j]) == ord("e") or (Int(s[j]) >= ord("0") and Int(s[j]) <= ord("9"))):
                    j += 1
                try:
                    out.append(atof(String(self.reply[byte=i:j])))
                except:
                    pass
                i = j - 1
            i += 1
        return out^

    def flag(self, key: String) -> Bool:
        return self.reply.find(String('"') + key + '": true') >= 0

    def held(self, fly: Int, input: String) -> Bool:
        return self.reply.find(String('{"fly": ') + String(fly) + ', "input": "' + input + '"') >= 0


def main() raises:
    var c = Ctl(String(argv()[1]))
    c.send(String(argv()[2]), String(argv()[3]) if len(argv()) > 3 else String())
    var t0 = perf_counter_ns()
    while len(c.pending) and perf_counter_ns() - t0 < 5_000_000_000:
        c.pump()
    print(c.reply)
