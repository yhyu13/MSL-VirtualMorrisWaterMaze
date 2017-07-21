import datetime
import struct
import logging
# noinspection PyUnresolvedReferencesa
import numpy as np
import os
import time
import re
import pytz
import scipy.spatial.distance as distance
from tzlocal import get_localzone


# This helper function extracts the meta-data from the filename
def get_filename_meta_data(fn):
    parts = fn.split('_')
    dt = datetime.datetime.strptime(parts[3] + '_' + parts[4].split('.')[0], '%Y-%m-%d_%H-%M-%S')
    return {"subid": parts[0], "trial": parts[1], "iteration": parts[2], "datetime": dt}


# Lookup phase name from number
def trial_num_to_str(trial):
    names = ['Test Trial', 'Practice - Flags', 'Practice - Hills', 'Practice - Visible Platform', 'Trial 1',
             'Trial 2-5', 'Trial 6-10', 'Trial 11-15', 'Probe Trial']
    #print(trial)
    lookup = int(trial)
    # noinspection PyTypeChecker
    #if isinstance(lookup, basestring):
    #    lookup = int(lookup)
    return names[0]


# From http://stackoverflow.com/questions/1550560/encoding-an-integer-in-7-bit-format-of-c-sharp-binaryreader-readstring
# This function is used in reading the binary files to read the length of the header from the beginning of the file
def decode_7bit_int_length(fp):
    string_length = 0
    string_length_parsed = False
    step = 0
    while not string_length_parsed:
        part = ord(fp.read(1))
        string_length_parsed = ((part >> 7) == 0)
        part_cutter = part & 127
        to_add = part_cutter << (step * 7)
        string_length += to_add
        step += 1
    return string_length


# From http://stackoverflow.com/questions/15919598/serialize-datetime-as-binary
# This function is used in reading the binary files to parse the binary .NET DateTime into a Python datetime
def datetime_from_dot_net_binary(data):
    kind = (data % 2 ** 64) >> 62  # This says about UTC and stuff...
    ticks = data & 0x3FFFFFFFFFFFFFFF
    seconds = ticks / 10000000
    tz = pytz.utc
    if kind == 0:
        tz = get_localzone()
    return datetime.datetime(1, 1, 1, tzinfo=tz) + datetime.timedelta(seconds=seconds)


def read_binary_file(path):
    iterations = []
    with open(path, 'rb') as f:
        header_length = decode_7bit_int_length(f)
        header = f.read(header_length)
        split_header = header.split(b',')

        version_number = split_header[1]

        num_keys = header.count(b'key')
        num_buttons = header.count(b'button')

        while f.read(1):  # Look ahead for end of file
            f.seek(-1, 1)  # Go back one to undo the look-ahead

            # Extract time information
            date_time = datetime_from_dot_net_binary(struct.unpack_from('q', f.read(8))[0])
            t = struct.unpack_from('f', f.read(4))[0]

            # Extract position information
            x = struct.unpack_from('f', f.read(4))[0]
            y = struct.unpack_from('f', f.read(4))[0]
            z = struct.unpack_from('f', f.read(4))[0]

            # Extract rotation information
            rx = struct.unpack_from('f', f.read(4))[0]
            ry = struct.unpack_from('f', f.read(4))[0]
            rz = struct.unpack_from('f', f.read(4))[0]
            rw = struct.unpack_from('f', f.read(4))[0]

            # Extract key, button, and item information according to expected numbers of each
            keys = []
            # noinspection PyRedeclaration
            for i in range(0, num_keys):
                keys.append(struct.unpack_from('?', f.read(1))[0])
            buttons = []
            # noinspection PyRedeclaration
            for i in range(0, num_buttons):
                buttons.append(struct.unpack_from('?', f.read(1))[0])

            # Store all information in simple dictionary and add to list of iterations
            iterations.append({"version": version_number,
                               "datetime": date_time, "time": t, "x": x, "y": y, "z": z,
                               "rx": rx, "ry": ry, "rz": rz, "rw": rw,
                               "keys": keys, "buttons": buttons})

        return iterations


def find_last(lst, sought_elt):
    for r_idx, elt in enumerate(reversed(lst)):
        if elt == sought_elt:
            return len(lst) - 1 - r_idx


def find_data_files_in_directory(directory, file_regex=""):
    if not os.path.exists(directory):
        raise IOError('The input path was not found.')

    start_time = time.time()
    data_files = []
    file_index = []
    file_roots_index = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            file_index.append(f)
            file_roots_index.append(root)

    regex = re.compile(file_regex)
    for root, f in zip(file_roots_index, file_index):
            if regex.search(os.path.basename(f)):
                logging.debug('Found data file ({0}).'.format(f))
                data_files.append(os.path.join(root, f))
    logging.info('Found {0} data files in {1} seconds.'.format(len(data_files), time.time() - start_time))
    return data_files


def get_exploration_metrics(iterations):
    total_time = (iterations[-1]['datetime'] - iterations[0]['datetime']).total_seconds()
    space_travelled = 0
    time_travelled = 0
    space_time_travelled = 0
    for idx, i in enumerate(iterations):
        if idx == len(iterations) - 1:
            break
        t = iterations[idx]['time']
        xy = [iterations[idx]['x'], iterations[idx]['y']]
        xyt = xy + [t]
        t_next = iterations[idx + 1]['time']
        xy_next = [iterations[idx + 1]['x'], iterations[idx + 1]['y']]
        xyt_next = xy_next + [t_next]
        space_travelled += distance.euclidean(xy, xy_next)
        space_time_travelled += distance.euclidean(xyt, xyt_next)
        time_travelled += distance.euclidean(t, t_next)

    return total_time, space_travelled, time_travelled, space_time_travelled
