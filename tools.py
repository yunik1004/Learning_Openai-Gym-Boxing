# Python batteries
import time


# Current time with microsecond unit
def current_time_microsecond():
    return int(time.time() * 1000000)
#end