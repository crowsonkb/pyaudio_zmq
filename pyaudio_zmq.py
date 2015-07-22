#!/usr/bin/env python3

import argparse
from collections import namedtuple
import logging
import numpy as np
import os
import pyaudio
import queue
import sys
import threading
import time
import zmq
import zmq.utils.jsonapi as jsonapi

ChunkMeta = namedtuple('ChunkMeta',
                       ['seq', 'chunksize', 'channels', 'dtype', 'srate',
                       'adc_time', 'input_call_time'])

logging.basicConfig(format='%(name)s.%(levelname)s:  %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__).rpartition('.')[0])

ctx = zmq.Context()
pa = pyaudio.PyAudio()

class AudioStream:
    def __init__(self, device, endpoint, channels, chunksize, srate):
        self.device = device
        self.endpoint = endpoint
        self.channels = channels
        self.chunksize = chunksize
        self.srate = srate
        self.stream = None
        self.logger = None
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
        self.logger.info('Audio device %d opened for %s', self.device, in_out)
        if in_out == 'input':
            latency = self.stream.get_input_latency()
        elif in_out == 'output':
            latency = self.stream.get_output_latency()
        self.logger.info('Audio stream reported %s latency: %.3f ms (%.1f samples)',
                         in_out, latency*1000, latency*self.srate)

class AudioInput(AudioStream):
    def __init__(self, *args, **_):
        super().__init__(*args)
        self.logger = logging.getLogger(logger.name+'.'+self.__class__.__name__)
        self.chunk_seq = 0
        self._open_stream('input')
        self.sock = ctx.socket(zmq.PUB)
        self.sock.bind(self.endpoint)
        self.logger.info('Bound to endpoint %s', self.endpoint)

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
        super().__init__(*args)

        self.latency_log = None
        if 'latency_debug' in kwargs and kwargs['latency_debug']:
            self.latency_log = list()
            self.callback_q = queue.Queue()
            self.callback_th = threading.Thread(
                target=self._callback_runner, name='AudioOutput.Callbacks',
                args=(self.callback_q,), daemon=True)
            self.callback_th.start()

        self.logger = logging.getLogger(logger.name+'.'+self.__class__.__name__)
        self.sock = ctx.socket(zmq.SUB)
        self.sock.subscribe = b''
        self.sock.connect(self.endpoint)
        self.logger.info('Connecting to endpoint %s', self.endpoint)

        msg = self.sock.recv_multipart()
        meta = ChunkMeta(*jsonapi.loads(msg[0]))
        self.logger.info(
            'Detected remote format: chunksize=%d channels=%d srate=%d',
            meta.chunksize, meta.channels, meta.srate)
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

        if self.latency_log is not None:
            self.latency_log.append(time_info['output_buffer_dac_time'] - meta.adc_time)
            if len(self.latency_log)*self.chunksize >= self.srate*5:
                self.callback_q.put(
                    (self._latency_report, (self.latency_log,), dict()))
                self.latency_log = list()
        data = np.frombuffer(msg[1], dtype=meta.dtype).reshape((-1, meta.channels))
        return (data.astype(np.float32), 0)

    def _callback_runner(self, q):
        while True:
            fn, args, kwargs = q.get()
            fn(*args, **kwargs)

    def _latency_report(self, log):
        self.logger.debug('It is now: %.9f s', time.monotonic())
        arr = np.array(log)
        lmax = np.max(arr)
        self.logger.debug('Total latency: - Max: %.3f ms (%.1f samples)',
                          lmax*1000, lmax*self.srate)
        lmed = np.median(arr)
        self.logger.debug('               - Med: %.3f ms (%.1f samples)',
                          lmed*1000, lmed*self.srate)
        lmin = np.min(arr)
        self.logger.debug('               - Min: %.3f ms (%.1f samples)',
                          lmin*1000, lmin*self.srate)


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
    parser.add_argument(
        '--channels', type=int, default=2,
        help='the number of audio channels to read')
    parser.add_argument(
        '--chunksize', type=int, default=256,
        help='the number of frames to read at once')
    parser.add_argument(
        '--srate', type=int, default=48000,
        help='the sample rate to read')
    parser.add_argument(
        '--latency-debug', action='store_true',
        help='print latency debugging statistics (audio outputs only)')
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
            stream = iam(device, args.endpoint, args.channels, args.chunksize, args.srate,
                         latency_debug=args.latency_debug)
            stream.remake.wait()
            logger.info('Recreating audio stream')
    except KeyboardInterrupt:
        print()

if __name__ == '__main__':
    main()
