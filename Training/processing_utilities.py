import typing
import torch
from Training import constants


class Event:
    def __init__(self, attribute_names: typing.List[str], values: typing.List, time_name, type_name):
        self.attributes = dict(zip(attribute_names, values))
        self.time_name = time_name
        self.type_name = type_name
        self.real_time = None

    def __getattr__(self, item):
        """
        "Simulate" a normal class
        :param item: attriute name to return
        :return:
        """
        if item == 'start_time' or item == 'end_time':
            return self.get_time()
        return self.attributes[item]

    def get_time(self):
        return self.attributes[self.time_name]

    def get_type(self):
        return self.attributes[self.type_name]

    @staticmethod
    def same_events(event1, event2) -> bool:
        return event1.attributes == event2.attributes

    def __len__(self):
        return len(self.attributes.keys())

    def __str__(self):
        if self.real_time is not None:
            self.attributes[self.time_name] = self.real_time
        join = ','.join(str(value) for value in self.attributes.values())
        return join

    def set_time_to_counter(self, counter, real_time):
        self.attributes[self.time_name] = counter
        self.real_time = real_time


def get_event_from_str(event_string, attribute_names, time_index, type_index):
    def convert_value(value: str):
        def isfloat(val: str):
            try:
                float(val)
                return True
            except ValueError:
                return False

        if str.isdigit(value):
            return int(value)
        if isfloat(value):
            return float(value)
        return value

    values = event_string.split(',') if event_string != "P" else [str(0)]*(4 + len(constants.event_format))
    for i, value in enumerate(values):
        values[i] = convert_value(value)
    time_name = attribute_names[time_index]
    type_name = attribute_names[type_index]
    new_event = Event(attribute_names, values, time_name, type_name)
    return new_event


def convert_event(event: Event):
    type_to_vec = {'A': [0, 0, 0, 1], 'B': [0, 0, 1, 0], 'C': [0, 1, 0, 0], 'D': [1, 0, 0, 0], 0: [0, 0, 0, 0]}
    values = list(event.attributes.values())
    type = type_to_vec[event.get_type()]
    values = type + [values[1]]
    return torch.tensor(values, dtype=torch.float32)