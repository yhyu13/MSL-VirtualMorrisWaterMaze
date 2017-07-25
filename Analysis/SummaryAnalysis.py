import argparse
import logging
import os
import scipy.spatial.distance as distance
import easygui
import time

import MSL_VMWM_BinaryReader as Log_Reader

########################################################################################################################
# Setup
########################################################################################################################

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

# Example Filename
# "C:\Users\Kevin\Documents\GitHub\Holodeck-Time-Travel-Task-Analytics\Visualizer\001_1_1_1_2016-08-29_10-26-03.dat"

# Parse Arguments
parser = argparse.ArgumentParser(description='When called without arguments, a directory dialog will appear to select '
                                             'a directory. When called with arguments, a directory can be selected '
                                             'via the --search_path argument. All files in the directory and '
                                             'subdirectories matching the \'..._\d_\d_YYYY-MM-DD_HH-mm-ss.dat\' '
                                             'standard regex for log files.')
parser.add_argument('--search_path', dest='search_path',
                    help='string path to a directory containing log files matching the search_regex')
parser.add_argument('--search_regex', dest='search_regex',
                    default='..._\d_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat',
                    help='a regular expression for selecting files in search_path')


def xycoordinate(s):
    try:
        _x, _y = map(float, s.split(','))
        return _x, _y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")


parser.add_argument('--next_to_point', dest='next_to_point', type=xycoordinate, default='5,-7',
                    help='the point from which to measure a special time total given the next_to_distance')
parser.add_argument('--next_to_distance', dest='next_to_distance',
                    default=9,
                    help='the euclidean distance within which a special time total will be computed from the '
                         'specified point')
parser.add_argument('--save_xyz_file', dest='save_xyz_file', default=True,
                    help='if True, a file *_xyz.* will be saved for each valid input file')
args = parser.parse_args()

save_xyz_file = args.save_xyz_file
next_to_distance = args.next_to_distance
point = args.next_to_point
regex = args.search_regex

# Get Log File Path and Load File
if args.search_path is None:
    directory = easygui.diropenbox()
else:
    directory = args.search_path
if directory is '':
    logging.info('No directory selected. Closing.')
    exit()
if not os.path.exists(directory):
    logging.error('Directory not found. Closing.')
    exit()

# Get file listing
files = Log_Reader.find_data_files_in_directory(directory, file_regex=regex)

summary_file = 'VMWM_Summary_' + time.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'

logging.info('Writing summary output to {0}.'.format(summary_file))

header = 'subid,trial,iteration,datetime,filepath,point,distance,total_travel_time,total_travel_distance,' \
         'cumulative_time_spent_next_to_point'

with open(summary_file, 'wb') as sfp:
    sfp.write(header + '\r\n')
    for f in files:
        # noinspection PyBroadException
        try:
            # The meta filename information for convenience
            meta = Log_Reader.get_filename_meta_data(os.path.basename(f))
        except:
            logging.error('There was an error reading the filename meta-information. Please confirm {0} is a valid log '
                          'file.'.format(f))
            exit()

        logging.info("Parsing file (" + str(f) + ")...")
        # First we populate a list of each iteration's data
        # This section of code contains some custom binary parser data which won't be explained here
        iterations = Log_Reader.read_binary_file(f)
        # Output the iterations count for debugging purposes

        x = []
        y = []
        t = []
        total_travel_time = 0
        total_travel_distance = 0
        cumulative_time_spent_next_to_point = 0
        for idx, i in enumerate(iterations):
            x.append(float(i['x']))
            y.append(float(i['z']))
            t.append(float(i['time']))
            if len(t) >= 2:
                time_interval = distance.euclidean(t[-1], t[-2])
                total_travel_time += time_interval
                total_travel_distance += distance.euclidean((x[-1], y[-1]), (x[-2], y[-2]))
                if distance.euclidean(point, (x[-1], y[-1])) <= next_to_distance:
                    cumulative_time_spent_next_to_point += time_interval

        line = ','.join([str(_m) for _m in [meta["subid"], meta["trial"], meta["iteration"], meta["datetime"],
                                            f, ' '.join([str(_p) for _p in point]), next_to_distance,
                                            total_travel_time, total_travel_distance,
                                            cumulative_time_spent_next_to_point]])
        sfp.write(line + '\r\n')

        if save_xyz_file:
            out_path = os.path.join(os.path.dirname(f),
                                    'xyz_' + os.path.splitext(os.path.basename(f))[0] +
                                    ''.join(os.path.splitext(os.path.basename(f))[1:]))
            logging.info('Saving xyz file to {0} with {1} points.'.format(out_path, len(t)))
            with open(out_path, 'wb') as fp:
                fp.writelines(['{0},{1},{2}\r\n'.format(xx, yy, tt) for xx, yy, tt in zip(x, y, t)])
