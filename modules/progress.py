class Bar:
    def __init__(self, steps, width=80):
        self.steps = steps
        self.progress = 0
        self.display_progress = 0
        self.display_steps = width - 2
        print('[' + self.display_steps * ' ' + ']', end='')

    def increment(self):
        self.progress += 1
        display_progress_update = round(self.display_steps * self.progress / self.steps)
        if self.display_progress < display_progress_update:
            self.display_progress = display_progress_update
            print("\r[" + self.display_progress * '#' + (self.display_steps - self.display_progress) * '.' + ']', end='')
        if self.progress == self.steps:
            print("")

    def __call__(self):
        self.increment()

if __name__ == "__main__":
    from time import sleep
    bar = Bar(100, width=80)
    for i in range(100):
        sleep(1/32)
        bar()
