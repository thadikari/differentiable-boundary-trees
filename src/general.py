import time, pickle


fetch_field = lambda lst, arg: list(x[arg] for x in lst)
is_field = lambda lst, arg: arg in lst[0]
dump_pik = lambda path, data: pickle.dump(data, open(path, 'wb'))
load_pik = lambda path: pickle.load(open(path, 'rb'))

    
def load_pik_try(path):
    try:
        return load_pik(path)
    except:
        return None
        

def time_id():
    t = time.mktime(time.gmtime())
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(t))

    
class TimeTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.timestamp = time.time()
    
    def elapsed(self, duration_sec, reset_if_true=False):
        ret = (time.time() - self.timestamp) > duration_sec
        if ret & reset_if_true:
            self.reset()
        return ret