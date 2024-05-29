# original idea:
# https://github.com/lanpa/tensorboard-dumper/blob/master/dump.py

import struct
import tensorboard.compat.proto.event_pb2 as event_pb2
import io
import argparse
from PIL import Image

from collections import defaultdict
from pathlib import Path

def read(data):
    header = struct.unpack('Q', data[:8])

    # crc_hdr = struct.unpack('I', data[:4])
    
    event_str = data[12:12+int(header[0])] # 8+4
    data = data[12+int(header[0])+4:]
    return data, event_str

    # crc_ev = struct.unpack('>I', data[:4])
    


parser = argparse.ArgumentParser(description='tensorboard-dumper')
parser.add_argument('--gif', default=False, action='store_true', help='save result as gif')
parser.add_argument('--csv', default=True, action='store_true', help='save result as csv')
parser.add_argument('--input', help='saved tensorboard file to read from')
parser.add_argument('--inputs', nargs='+', default=[], help='multiple inputs')
parser.add_argument('--output', default='output.gif', help='output filename for gif export')
parser.add_argument('--maxframe', type=int, default=100, help='limit the number of frames')
parser.add_argument('--duration', type=int, default=100, help='show time for each frame (ms)')

args = parser.parse_args()

if args.input:
    args.inputs.insert(0, args.input)
if not args.inputs:
    args.inputs = ['demo.pb']

args.inputs = [Path(i) for i in args.inputs]


def do_one_input(input_fpath):

    try:
        with open(input_fpath, 'rb') as f:
            data = f.read()
    except FileNotFoundError:
        print('input file not found')
        exit()
    except IsADirectoryError:
        print("Is a directory, falling back to the first file that looks like a Tensorboard log...")
        return do_one_input(next(input_fpath.glob("events.out.tfevents.*")))

    images = []
    simple_values = defaultdict(list)

    while data and args.maxframe>0:
        args.maxframe = args.maxframe-1
        data, event_str = read(data)
        event = event_pb2.Event()

        event.ParseFromString(event_str)
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    simple_values[value.tag].append([event.step, value.simple_value])
                    print(value.simple_value, value.tag, event.step)
                if value.HasField('image'):
                    # img = value.image
                    # img = Image.open(io.BytesIO(img.encoded_image_string))
                    # if args.gif:
                    #     images.append(img)
                    # else:
                    #     img.save('img_{}.png'.format(event.step), format='png')
                    print('img (not) saved.')

    if args.csv:
        for tag in simple_values:
            fpath = input_fpath.parent / (tag.replace("/", "-")+".csv")
            with open(fpath, "w") as f:
                f.write("step,value\n")
                f.writelines(str(step) + "," + str(value) + "\n" for step,value in simple_values[tag])
            print(f"saving {fpath}")

    if args.gif:
        from PIL import Image
        im = images[0]
        im.save(args.output, save_all=True, append_images=images, duration=100, loop=0) # forever

for inp in args.inputs:
    do_one_input(inp)
