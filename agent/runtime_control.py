"""Same-user local control, acknowledged only by the runner at exchange boundaries."""
import json
import os
from pathlib import Path
import re
import socket
import struct
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / '.work/neural-flies/control-sessions'
ACTIONS = ('status', 'pause', 'resume', 'reset_body_brain', 'step', 'stop', 'checkpoint', 'world_edit', 'drive', 'camera')
EDITS = ('world_edit', 'drive', 'camera')  # actions that carry an `edit` object (drive, camera: engine runners, agent/body_engine/engine_control.py)


def peer_is_owner(sock):
    return struct.unpack('3i', sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1] == os.getuid()


class RuntimeControl:
    def __init__(self, out, profile, backend):
        self.session_id = uuid.uuid4().hex
        self.address = '\0fly-control-' + self.session_id
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.socket.bind(self.address)
        self.socket.listen(8)
        self.socket.setblocking(False)
        self.waiting_step = None
        REGISTRY.mkdir(parents=True, exist_ok=True)
        self.path = REGISTRY / (self.session_id + '.json')
        self.record = {'session_id': self.session_id, 'pid': os.getpid(), 'uid': os.getuid(),
            'output': str(Path(out).resolve()), 'profile': profile, 'backend': backend,
            'created_unix': time.time(), 'lifecycle': 'running'}
        self.publish()
        (Path(out) / 'control-session.json').write_text(json.dumps(self.record, indent=2) + '\n')

    def publish(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.record, indent=2) + '\n')
        temporary.replace(self.path)

    def poll(self):
        requests = []
        for _ in range(8):
            try: connection, _ = self.socket.accept()
            except BlockingIOError: break
            connection.settimeout(.1)
            try:
                if not peer_is_owner(connection): raise ValueError('peer uid mismatch')
                raw = connection.recv(4097)
                if len(raw) > 4096: raise ValueError('request too large')
                request = json.loads(raw)
                if not isinstance(request, dict): raise ValueError('request must be an object')
                if not re.fullmatch('[0-9a-f]{32}', str(request.get('request_id',''))): raise ValueError('invalid request_id')
                if request.get('session_id') != self.session_id: raise ValueError('session_id mismatch')
                if request.get('action') not in ACTIONS: raise ValueError('unsupported action')
                steps = request.get('steps', 1)
                if type(steps) is not int or not 1 <= steps <= 100: raise ValueError('steps must be integer in [1,100]')
                if request['action'] != 'step' and steps != 1: raise ValueError('steps applies only to step')
                if (request['action'] in EDITS) != isinstance(request.get('edit'), dict): raise ValueError('edit object goes with ' + ', '.join(EDITS) + ' only')
                requests.append((connection, request))
            except (ValueError, OSError) as exc:
                self.reply(connection, {'status': 'REJECTED', 'error': str(exc)})
        return requests

    def reply(self, connection, answer):
        try: connection.send(json.dumps(dict(answer, session_id=self.session_id)).encode())
        except OSError: pass  # A disconnected client does not undo an applied action.
        finally: connection.close()

    def finish_step(self, answer):
        if self.waiting_step:
            connection, request = self.waiting_step
            self.reply(connection, dict(answer, action='step', request_id=request['request_id']))
            self.waiting_step = None

    def close(self, error=None):
        self.finish_step({'status': 'ABORTED', 'error': error or 'runner stopped before step completed'})
        self.socket.close()
        self.record.update(lifecycle='failed' if error else 'stopped', error=error, finished_unix=time.time())
        self.publish()


def session_records():
    return [json.loads(path.read_text()) for path in sorted(REGISTRY.glob('*.json'))]


def request_control(session_id, action, steps=1, timeout=120, edit=None):
    if not isinstance(session_id, str) or not re.fullmatch('[0-9a-f]{32}', session_id):
        raise ValueError('expected explicit 32-character session_id from fly_status')
    if action not in ACTIONS: raise ValueError('unsupported action')
    if type(steps) is not int or not 1 <= steps <= 100: raise ValueError('steps must be integer in [1,100]')
    record = json.loads((REGISTRY / (session_id + '.json')).read_text())
    if record['lifecycle'] != 'running' or record['uid'] != os.getuid():
        raise ValueError('session is stopped or belongs to another user')
    request_id = uuid.uuid4().hex
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.settimeout(timeout)
        connection.connect('\0fly-control-' + session_id)
        if not peer_is_owner(connection): raise ValueError('server uid mismatch')
        connection.send(json.dumps({'session_id': session_id, 'request_id': request_id,
            'action': action, 'steps': steps, **({'edit': edit} if edit is not None else {})}).encode())
        try: raw = connection.recv(65536)
        except TimeoutError as exc:
            raise TimeoutError('No acknowledgement; action may have applied. Query status before retrying.') from exc
    answer = json.loads(raw)
    if answer.get('session_id') != session_id: raise ValueError('response session_id mismatch')
    if answer.get('status') != 'APPLIED': raise RuntimeError(json.dumps(answer))
    if answer.get('request_id') != request_id: raise ValueError('response request_id mismatch')
    return answer
