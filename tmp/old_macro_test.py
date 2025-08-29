import types

class MyCallable:
    def __init__(self, func):
        # self.on_Call = func
        pass

    def __call__(self, *args, **kwargs):
        return self.on_Call(*args, **kwargs)


# Define different call behaviors
def call_behavior_1(*args, **kwargs):
    print("Call behavior 1", args, kwargs)

def call_behavior_2(*args, **kwargs):
    print("Call behavior 2", args, kwargs)

# Create instances with different call methods
a = MyCallable(call_behavior_1)
b = MyCallable(call_behavior_2)

a(1, 2, x=10)  # Output: Call behavior 1 (1, 2) {'x': 10}
b(3, 4, y=20)  # Output: Call behavior 2 (3, 4) {'y': 20}
