"""
A tiny module for creating a progress bar as a class instance.
This progress bar has a set number of steps defined at creation and a set length
in characters. It can be updated only one step at a time, and will rewrite the
last line at every increment call.
"""
class Bar:
    def __init__(self, steps, width=40):
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
        if self.progress == self.steps - 1:
            print("")

    def __call__(self):
        self.increment()

if __name__ == "__main__":
    from time import sleep
    bar = Bar(100)
    for i in range(100):
        sleep(1/32)
        bar()
