import numpy as np


def read_textfile(filename, step=0):
    """
    Read a textfile and return a list of lines
    :param filename: the path to the textfile
    :param step: the number of lines per example (for example, if step=4, then every two lines are grouped together)
    :return: a list of lines
    """
    with open(filename, "r") as f:
        textfile = [line.strip() for line in f.readlines()]

    # reshape to 2D array
    if step:
        textfile = np.array(textfile).reshape((-1, step)).tolist()

    return textfile


def cont_write_textfile(data, output_path):
    """
    Write data to a textfile in a continuous manner
    :param data: a list of lines
    :param output_path: the path to the output textfile
    :return: None
    """
    with open(output_path, "a", encoding='utf-8') as f:
        for line in data:
            f.write(line)
            f.write("\n")
