import json


class SourceFile():

    def __init__(self, name, method):
        self.name = name
        self.method = method

    def read(self):
        with open(self._file_name) as data_file:
            data = json.load(data_file)
        return data

    def write(self, data):
        with open(self._file_name, 'w') as outfile:
            json.dump(data, outfile)    

    @property
    def _file_name(self):
        return 'data/' + self.name + '_' + self.method + '.json'
