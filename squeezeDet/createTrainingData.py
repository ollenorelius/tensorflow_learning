"""
Simple script to create training data.

Command line args are <folder> <count>.
"""
import creator as c
import sys

if len(sys.argv) != 3:
    print('Simple script to create training data. '
          + 'Command line args are <folder> <count>.')
else:
    c.save_justOne_batch_to_disk(sys.argv[1], int(sys.argv[2]))
