from langchain_core.runnables import Runnable
from langchain_core.load import dumpd, load

class FakeGenerator(Runnable):
    def invoke(self, input, config=None, **kwargs):
        # Ignore input and return a fixed answer
        return "The answer is apples."
    
    def batch(self, inputs, config=None, **kwargs):
        return [self.invoke(input, config, **kwargs) for input in inputs]
    
    def stream(self, input, config=None, **kwargs):
        yield self.invoke(input, config, **kwargs)
