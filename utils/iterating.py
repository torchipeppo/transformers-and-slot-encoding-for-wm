# an infinitely looping iterator
class infiniter():
    def __init__(self, iterable):
        self.iterable=iterable
        self.it = iter(iterable)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return self.it.next()
        except StopIteration:
            self.it = iter(self.iterable)
            return self.it.next()
    def next(self):
        return self.__next__()