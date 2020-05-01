# Import everything from pycc.lib
# I don't understand -- isn't that this very file?
from pycc.lib import *
# Where is pycctx?
# From that, import Tx -- a custom module to describe a Tx, I suppose
from pycctx import Tx


# Definition of the CCApp class
class CCApp:

    # Constructor
    def __init__(self, schema, eval_code, chain):
        self.schema = schema
        self.eval_code = eval_code
        self.chain = chain

    # The __call__ function allows for the CCApp class to behave like a function, being called and provided with params
    def __call__(self, *args, **kwargs):
        # When called, it provides the cc_eval of the args and kwargs
        return self.cc_eval(*args, **kwargs)

    # Evaluate/validate the chain and something decoded from a bin
    def cc_eval(self, tx_bin):
        self.validate_tx(chain, Tx.decode_bin(tx_bin))

    # Definition for the get_model() method
    def get_model(self, name):
        try:
            # Split the name into a module name and a model name
            (module_name, model_name) = name.split('.', 2)
            # Return the self's schema with a matrix array
            return self.schema[module_name][model_name]
        except:
            # Error handling
            raise AssertionError("Invalid model: %s" % name)

    # Validate a TX
    # Go TX -> Condensed
    def validate_tx(self, tx):
        return TxValidator(self, tx).validate()

    # Create a TX
    # Go Condensed -> TX
    def create_tx(self, spec):
        return TxConstructor(self, spec).construct()
