"""
Python thread pool, see
http://code.activestate.com/recipes/577187-python-thread-pool/
Author: Valentin Kuznetsov <vkuznet [AT] gmail [DOT] com>
"""
from builtins import range, object

# System modules
import time
import json
import hashlib
import logging
import threading
from queue import Queue


def genkey(query):
    """Generate a new key-hash for a given query. We use md5 hash for
    the query and key is just hex representation of this hash.
    """
    if  isinstance(query, dict):
        record = dict(query)
        query = json.JSONEncoder(sort_keys=True).encode(record)
    keyhash = hashlib.md5()
    keyhash.update(query.encode('utf-8', 'strict'))
    return keyhash.hexdigest()

def set_thread_name(ident, name):
    """Set thread name for given identified."""
    for thr in threading.enumerate():
        if  thr.ident == ident:
            thr.name = name
            break

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to
    check regularly for the stopped() condition.
    """
    def __init__(self, target, name, args):
        super(StoppableThread, self).__init__(
            target=target, name=name, args=args)
        self._stop_event = threading.Event()

    def stop(self):
        """Set event to stop the thread."""
        self._stop_event.set()

    def stopped(self):
        """Return stopped status of the thread."""
        return self._stop_event.is_set()

    def running(self):
        """Return running status of the thread."""
        return not self._stop_event.is_set()

def start_new_thread(name, func, args, unique=False):
    """Wrapper wroung standard thread.strart_new_thread call."""
    if unique:
        threads = sorted(threading.enumerate())
        for thr in threads:
            if name == thr.name:
                return thr
#     thr = threading.Thread(target=func, name=name, args=args)
    thr = StoppableThread(target=func, name=name, args=args)
    thr.daemon = True
    thr.start()
    return thr

class UidSet():
    """UID holder keeps track of uid frequency."""
    def __init__(self):
        self.set = {}

    def add(self, uid):
        """Add given uid or increment uid occurence in a set."""
        if not uid:
            return
        if uid in self.set:
            self.set[uid] += 1
        else:
            self.set[uid] = 1

    def discard(self, uid):
        """Either discard or downgrade uid occurence in a set."""
        if uid in self.set:
            self.set[uid] -= 1
        if uid in self.set and not self.set[uid]:
            del self.set[uid]

    def __contains__(self, uid):
        """Check if uid present in a set."""
        if uid in self.set:
            return True
        return False

    def get(self, uid):
        """Get value for given uid."""
        return self.set.get(uid, 0)

class Worker(threading.Thread):
    """Thread executing worker from a given tasks queue."""
    def __init__(self, name, taskq, pidq, uidq, logger=None):
        self.logger = logging.getLogger()
        threading.Thread.__init__(self, name=name)
        self.exit = 0
        self.tasks = taskq
        self.pids = pidq
        self.uids = uidq
        self.daemon = True
        self.start()

    def force_exit(self):
        """Force run loop to exit in a hard way."""
        self.exit = 1

    def run(self):
        """Run thread loop."""
        while True:
            if self.exit:
                return
            task = self.tasks.get()
            if task is None:
                return
            if self.exit:
                return
            if isinstance(task, str):
                print(f'Worker daemon run {task}')
            elif isinstance(task, tuple) and len(task) == 5:
                evt, pid, func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                    self.pids.discard(pid)
                except Exception as exc:
                    self.pids.discard(pid)
                    msg = f'func={func} args={args} kwargs={kwargs}'
                    self.logger.error(f'error {str(exc)}, call {msg}')
                evt.set()
            else:
                print(f'Unsupported task {task}')

class TaskManager():
    """Task manager class based on thread module which executes
    assigned tasks concurently. It uses a pool of thread workers,
    queue of tasks and pid set to monitor jobs execution.

    .. doctest::

        Use case:
        mgr  = TaskManager()
        jobs = []
        jobs.append(mgr.spawn(func, args))
        mgr.joinall(jobs)
    """
    def __init__(self, nworkers=10, name='TaskManager'):
        self.logger = logging.getLogger()
        self.name = name
        self.pids = set()
        self.uids = UidSet()
        self.tasks = Queue()
        self.workers = [Worker(name, self.tasks, self.pids, self.uids,
                               self.logger)
                        for _ in range(0, nworkers)]

    def status(self):
        """Return status of task manager queue."""
        info = {'qsize':self.tasks.qsize(), 'full':self.tasks.full(),
                'unfinished':self.tasks.unfinished_tasks,
                'nworkers':len(self.workers)}
        return {self.name: info}

    def nworkers(self):
        """Return number of workers associated with this manager."""
        return len(self.workers)

    def spawn(self, func, *args, **kwargs):
        """Spawn new process for given function."""
        pid = kwargs.get('pid', genkey(str(args) + str(kwargs)))
        evt = threading.Event()
        if  not pid in self.pids:
            self.pids.add(pid)
            task = (evt, pid, func, args, kwargs)
            self.tasks.put(task)
        else:
            # the event was not added to task list, invoke set()
            # to pass it in wait() call, see joinall
            evt.set()
        return evt, pid

    def remove(self, pid):
        """Remove pid and associative process from the queue."""
        self.pids.discard(pid)

    def is_alive(self, pid):
        """Check worker queue if given pid of the process is still
        running.
        """
        return pid in self.pids

    def clear(self, tasks):
        """Clear all tasks in a queue. It allows current jobs to run,
        but will block all new requests till workers event flag is set
        again.
        """
        # Each task is return from spawn, i.e. a pair (evt, pid)
        _ = [t[0].clear() for t in tasks]

    def joinall(self, tasks):
        """Join all tasks in a queue and quit."""
        # Each task is return from spawn, i.e. a pair (evt, pid)
        _ = [t[0].wait() for t in tasks]

    def quit(self):
        """Put None task to all workers and let them quit."""
        _ = [self.tasks.put(None) for _ in self.workers]
        # Let workers threads cool-off and quit
        time.sleep(1)
