# Recipe from Python Cookbook, 2.23 Reading an Unbuffered Character in a Cross-Platform Way
import sys

try:
    from msvcrt import getwch as getch
except ImportError:
    """We are not on Windows; try the Unix-like approach"""

    def getch():
        import sys, tty, termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


SKIP = False
STEPS_TO_SKIP = 0

def wait():
    global SKIP, STEPS_TO_SKIP
    if SKIP:
        return
    if STEPS_TO_SKIP > 0:
        STEPS_TO_SKIP -= 1
        return
    print("Press 'n' for next step, <num> to skip <num> steps, 'q' to quit, 's' to run to finish: ", end='')
    sys.stdout.flush()
    while True:
        key = getch().lower()
        print(key)
        if key == "n":
            break
        if key.isdigit():
            STEPS_TO_SKIP = int(key)
            break
        if key == "s":
            SKIP = True
            break
        if key == "q":
            sys.exit(1)
