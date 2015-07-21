#!/usr/bin/env python3

import argparse
from collections import namedtuple
import numpy as np
import pyaudio
import sys
import threading
import zmq
import zmq.utils.jsonapi as jsonapi

ChunkMeta = namedtuple('ChunkMeta',
                       ['seq', 'chunksize', 'channels', 'dtype', 'srate',
                       'adc_time', 'input_call_time'])

ctx = zmq.Context()
pa = pyaudio.PyAudio()

class AudioStream:
    def __init__(self, device, endpoint):
        self.device = device
        self.endpoint = endpoint
        self.channels = 2
        self.chunksize = 256
        self.srate = 48000
        self.stream = None
        self.remake = threading.Event()

    def __call__(self, _, __, ___, ____):
        raise NotImplementedError('Do not instantiate this class directly!')

    def _open_stream(self, in_out):
        stream_info = pyaudio.PaMacCoreStreamInfo(
            pyaudio.PaMacCoreStreamInfo.paMacCorePro)
        kwargs = dict(frames_per_buffer=self.chunksize, stream_callback=self)
        kwargs[in_out] = True
        kwargs['%s_device_index' % in_out] = self.device
        kwargs['%s_host_api_specific_stream_info' % in_out] = stream_info
        self.stream = pa.open(self.srate, self.channels, pyaudio.paFloat32,
                              **kwargs)

class AudioInput(AudioStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_seq = 0
        self._open_stream('input')
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(self.endpoint)

    def __call__(self, in_data, nframes, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32).reshape((-1, self.channels))
        meta = ChunkMeta(
            self.chunk_seq, data.shape[0], data.shape[1], str(data.dtype), self.srate,
            time_info['input_buffer_adc_time'], time_info['current_time'])
        self.sock.send_multipart([jsonapi.dumps(meta), data])
        self.chunk_seq += 1
        return (None, 0)

class AudioOutput(AudioStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sock = ctx.socket(zmq.SUB)
        self.sock.subscribe = b''
        self.sock.connect(self.endpoint)
        msg = self.sock.recv_multipart()
        meta = ChunkMeta(*jsonapi.loads(msg[0]))
        self.chunksize = meta.chunksize
        self.channels = meta.channels
        self.srate = meta.srate
        self._open_stream('output')

    def __call__(self, _, nframes, time_info, status):
        msg = self.sock.recv_multipart()
        if len(msg[1]) == 0:
            self.remake.set()
            return (b'', 2)
        meta = ChunkMeta(*jsonapi.loads(msg[0]))

        if (meta.chunksize != self.chunksize) or \
           (meta.channels != self.channels) or \
           (meta.srate != self.srate):
            self.remake.set()
            return (b'', 2)

        data = np.frombuffer(msg[1], dtype=meta.dtype).reshape((-1, meta.channels))
        return (data.astype(np.float32), 0)

def list_devices():
    n = pa.get_device_count()
    devices = [pa.get_device_info_by_index(i) for i in range(n)]
    for device in devices:
        i = device['index']; name = device['name']
        ch_in = device['maxInputChannels']; ch_out = device['maxOutputChannels']
        hz = device['defaultSampleRate']
        l_in = device['defaultLowInputLatency']; l_out = device['defaultLowOutputLatency']
        l_in_ms, l_out_ms = (l_in*1000, l_out*1000)
        l_in_hz, l_out_hz = (l_in*hz, l_out*hz)
        print('Device #%d: %s' % (i, name))
        print('%d input, %d output channels' % (ch_in, ch_out))
        print('%.0f Hz default sample rate' % hz)
        if ch_in > 0:
            print('%.3f ms input latency (%.1f samples)' % (l_in_ms, l_in_hz))
        if ch_out > 0:
            print('%.3f ms output latency (%.1f samples)' % (l_out_ms, l_out_hz))
        print()

def main():
    parser = argparse.ArgumentParser()
    in_out = parser.add_mutually_exclusive_group(required=True)
    in_out.add_argument(
        '-i', type=int, default=-1,
        help='read from specified audio device number')
    in_out.add_argument(
        '-o', type=int, default=-1,
        help='write to specified audio device number')
    in_out.add_argument(
        '-l', action='store_true', help='list audio devices on this system')
    parser.add_argument(
        'endpoint', nargs='?', help='the ZMQ endpoint to bind/connect to')
    args = parser.parse_args()

    if args.i >= 0:
        device = args.i
        iam = AudioInput
    elif args.o >= 0:
        device = args.o
        iam = AudioOutput
    elif args.l:
        list_devices()
        return
    else:
        raise NotImplementedError('This shouldn\'t happen.')

    if not args.endpoint:
        print('You must specify a ZMQ endpoint if you specified -i or -o.',
              file=sys.stderr)
        print('Examples: "ipc://socket.ipc", "tcp://127.0.0.1:5555"\n',
              file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            stream = iam(device, args.endpoint)
            stream.remake.wait()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
