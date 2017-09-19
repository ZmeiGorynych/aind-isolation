class Borg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state

class Reporting(Borg):
    def __init__(self):
        Borg.__init__(self)
        if 'report' not in self.__dict__:
            self.report = []


