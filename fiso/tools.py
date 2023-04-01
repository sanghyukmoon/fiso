import time

def timer(message="", verbose=True):
    """Measure wall clock time

    Arguments
    ---------
    message: string, optional
        Short description of the job to measure time
    verbose: boolean, optional
        if True, print out measured time

    Return
    ------
    Etot: float
        Total energy-like contained in the isocontour level
    """
    if message=="":
        time.prevtime = time.time()
        return
    thistime = time.time()
    dt = thistime - time.prevtime
    time.prevtime = thistime
    if len(message) > 0 and verbose:
        print("{}: {} seconds elapsed".format(message, dt))
    return dt
