import logging, logging.handlers, sys


# sys.stdout = fp
# sys.stderr = fp
#loggers = {}
__inited = False

def setup_logger(file='log.log', name=''):
    global __inited
    if not __inited:
        fp = open(file, 'w')
        logger = logging.getLogger('')
        logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler(fp)
        sh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s|%(message)s')
        sh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(ch)
        __inited = True
    return logging.getLogger(name)
#loggers[name] = logger