import multiprocessing as mp
import time


class TestClass:
    def __init__(self):
        self.q = mp.Queue()
        self.proc1 = mp.Process(target=self.func)
        self.proc1.start()

    def func(self):
        for i in range(10):
            self.q.put(["Hellow",1])
            time.sleep(0.5)

    def main(self):
        cnt = 0
        while True:
            try:
                e = self.q.get_nowait()
            except:
                e = None
            if not (e is None):
                print(e)
            time.sleep(0.01)
            cnt += 1
            if cnt >= 1500:
                self.proc1.start()

if __name__ == "__main__":
    t = TestClass()
    t.main()