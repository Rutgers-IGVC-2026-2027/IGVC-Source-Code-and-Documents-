#!/usr/bin/env python3
"""Small synthetic input generator. Classifications come only from C++ replay."""
import csv
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: generate_demo_csv.py output.csv')
    with open(sys.argv[1], 'w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['stamp', 'odom_x', 'odom_y', 'x', 'y', 'z'])
        for frame in range(20):
            stamp = 1 + frame * 0.1
            odom_x = -3 + frame * 0.3
            for ix in range(-40, 41):
                for iy in range(-1, 2):
                    x, y = ix * 0.2 + 0.05, iy * 0.2 + 0.05
                    for sample in range(6):
                        px = x + sample * 0.003
                        writer.writerow([stamp, odom_x, 0, px, y, 0.3 * px])
                    if 5 <= ix <= 9:
                        for sample in range(2):
                            writer.writerow([stamp, odom_x, 0, x, y, 0.3 * x + 0.7])
            if frame == 0:
                writer.writerow([stamp, odom_x, 0, 0.05, 1.05, 0])
        writer.writerow([6, odom_x, 0, '', '', ''])


if __name__ == '__main__':
    main()
