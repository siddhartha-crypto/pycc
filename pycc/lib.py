# The binascii module contains a number of methods to convert between binary and various ASCII-encoded binary representations
import binascii

# The io module provides Python’s main facilities for dealing with various types of I/O. There are three main types of I/O: text I/O, binary I/O and raw I/O
import io

# Json module
import json

# Assignment statements in Python do not copy objects, they create bindings between a target and an object. For collections that are mutable or contain mutable items, a copy is sometimes needed so one can change one copy without changing the other
import copy

# factory function for creating tuple subclasses with named fields
from collections import namedtuple

# From copy module above
from copy import deepcopy

# Custom Komodo code
from pycctx import *

# Hack because komodod expects cc_eval function and pycctx.script also exports it
mk_cc_eval = cc_eval
del globals()['cc_eval']

# A transaction constructor class
class TxConstructor:

    # Class initialization constructor
    def __init__(self, app, spec):
        # Name this after the provided app's model spec name
        self.model = app.get_model(spec['name'])

        # Within the TxConstructor object, set the app object to the provided app (not a pointer, I assume?)
        self.app = app

        # A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original
        self.spec = deepcopy(spec)

        # Empty params
        self.params = {}
        self.stack = [] # TODO: phase out in favour of params

    # Construct method
    def construct(self):
        def f(l):
            out = []
            groups = []
            specs = self.spec[l]
            model = self.model[l]
            # Check that the length of specs and model are the same
            assert len(specs) == len(model), ("number of %s groups differs" % l)

            # The zip() function creates an iterator that will aggregate elements from two or more iterables.
            for (spec, model_i) in zip(specs, model):
                r = model_i.construct(self, spec)
                n = len(r)
                assert n <= 0xff, ("%s group too large (255 max)" % l)
                groups.append(n)
                out.extend(r)
            return (groups, out)

        # Not sure what these do. I do know what an f string is, but I don't understand how it's being used.
        (input_groups, inputs) = f('inputs')
        (output_groups, outputs) = f('outputs')

        params = [self.spec['name'], (input_groups, output_groups), self.params] + self.stack
        outputs += [TxOut.op_return(encode_params(params))]

        # I don't conceptually understand what is going on here
        return Tx(
            inputs = tuple(inputs),
            outputs = tuple(outputs)
        )

    # Property creates setters, getters, and deleters for an object
    @property
    def inputs(self):
        return tuple(i if type(i) == list else [i] for i in self.spec['inputs'])

# Transaction Validator class
class TxValidator:

    # Constructor
    def __init__(self, app, tx):
        # Initiate the app and tx properties
        self.tx = tx
        self.app = app

        # Acquire OP_RETURN params, I assume?
        self.stack = decode_params(get_opret(tx))

        # The first OP_RETURN param must be the name?
        self.name = self.stack.pop(0)

        # Get the associated app model and set it for this model
        self.model = app.get_model(self.name)

        # Set up the input/output groups
        (self.input_groups, self.output_groups) = self.stack.pop(0)
        self.params = self.stack.pop(0)

    # The validate function
    def validate(self):
        # An object of descriptive properties and inputs/outputs
        spec = {"txid": self.tx.hash, "inputs": [], "outputs": [], "name": self.name}

        def f(groups, l, nodes):
            assert len(groups) == len(self.model[l])
            assert sum(groups) == len(nodes)

            # Zip is aggregating something here, but I am not sure what?
            for (n, m) in zip(groups, self.model[l]):
                spec[l].append(m.consume(self, nodes[:n]))
                nodes = nodes[n:]

        # I don't understand how the f() function here and above is functioning
        f(self.input_groups, 'inputs', self.tx.inputs)
        f(self.output_groups, 'outputs', self.tx.outputs[:-1])

        # Validate all the self/spec pairs ?
        for validate in self.model.get('validators', []):
            validate(self, spec)

        # Return the resulting spec (what is specifically in here? Specification?)
        return spec

    # Definition for the get_input_group method
    def get_input_group(self, idx):

        # Grab the input_groups from this class instance
        groups = self.input_groups

        # Check that the length of the idx param is less than the groups' length
        assert idx < len(groups), "TODO better message"
        # Sum of all groups' idx values
        skip = sum(groups[:idx])
        #Return sum of transaction inputs
        return self.tx.inputs[skip:][:groups[idx]]

    # Definition for the get_output_group method
    def get_output_group(self, idx):
        groups = self.output_groups
        assert idx < len(groups), "TODO better message"
        # Sum of all groups' idx values
        skip = sum(groups[:idx])
        #Return sum of transaction outputs
        return self.tx.outputs[skip:][:groups[idx]]

    # Definition for get_group_for_output
    def get_group_for_output(self, n_out):
        groups = self.output_groups
        tot = 0
        # Scan for the requested n_out's out_m and return
        for (out_m, n) in zip(self.model['outputs'], groups):
            if tot + n > n_out:
                return out_m
            tot += n
        raise AssertionError("Cannot get group for output")


# Convert python data to hex value
def py_to_hex(data):
    return hex_encode(json.dumps(data, sort_keys=True))

# Encode provided data to hex value
def hex_encode(data):
    if hasattr(data, 'encode'):
        data = data.encode()
    return binascii.hexlify(data).decode()

# Decode hex value
def hex_decode(data):
    return binascii.unhexlify(data)

# Definition for the get_opret method
def get_opret(tx):
    # Ensure there are outputs, first
    assert tx.outputs, "opret not present"
    opret = tx.outputs[-1]
    assert opret.amount == 0
    # Get the opreturn data of a transaction output
    data = opret.script.get_opret_data()
    assert not data is None, "opret not present"
    return data


# Encode params (into what?)
def encode_params(params):
    return repr(params).encode()

# Decode params (from an eval statement? Is this a carryover from CC?)
def decode_params(b):
    return eval(b)


# Definition for the Input class
class Input:
    # Constructor
    def __init__(self, script):

        # Set provided script to self's script property
        self.script = script

    # Definition for the Input class consume method
    def consume(self, tx, inputs):
        # Check that there's only one input
        assert len(inputs) == 1
        # Set he input to the provided transaction
        return self.consume_input(tx, *inputs)

    # Definition for the consume_input method
    def consume_input(self, tx, inp):
        # Sets the following two json properties within the Input class instance
        return {
            "previous_output": inp.previous_output,
            "script": self.script.consume_input(tx, inp) or {}
        }

    # Construct the input from the provided spec
    def construct(self, tx, spec):
        return [self.construct_input(tx, spec)]

    # More construction of the provided input
    def construct_input(self, tx, spec):
        return TxIn(spec['previous_output'], self.script.construct_input(tx, spec.get('script', {})))


# Definition for the Inputs class
class Inputs:
    # Constructor
    def __init__(self, script, min=1):
        self.script = script
        # Requires at least one Input class instance, or more
        self.min = min

    # Consume the provided inputs
    def consume(self, tx, inputs):
        # Check that there's the minimum number of inputs in the provided inputs object
        # I assume having more than one input can be useful in coding
        assert len(inputs) >= self.min
        # Creating an Input instance from the script in this Inputs class instance
        inp = Input(self.script)
        # Return some type of script to consume the input for each input in inputs, providing the transaction in this consume() method
        return [inp.consume_input(tx, i) for i in inputs]

    # Definition for the construct method
    def construct(self, tx, inputs):
        # Check that the length of the inputs meets the minimum requirement
        assert len(inputs) >= self.min
        # Create an Input instance using the provided script
        i = Input(self.script)
        # Again, return some type of script that goes through the inputs and executes the construct_input() method for each input
        return [i.construct_input(tx, inp) for inp in inputs]


# Definition of the Outputs class
class Outputs:
    # Constructor
    # By default, the outputs class has no output amounts, there needs to be at least one output, the max I'm not sure (something hex related, possibly FF or 255 as an integer, making for a range of 1 to 255), and by default there's no data
    def __init__(self, script, amount=None, min=1, max=0xff, data=None):
        self.script = script
        self.amount = amount or Amount()
        self.data = data or {}
        self.min = min
        self.max = max

    # Definition for the consume output method
    def consume(self, tx, outputs):
        # Check that for the outputs object provided, it is within the appropriate range
        assert self.min <= len(outputs) <= self.max
        # Caputre all the outputs and store in a outs variable relative to the Outputs class
        outs = [self._consume_output(tx, o) for o in outputs]
        # For each output in the outs outputs object, capture the index number and store it as a part of the outputs
        # Capture all the transactions and transaction parameters
        # There's something in here about a member function for the data member variable as well
        # Not sure if that's recursive
        for out in outs:
            for (i, k) in enumerate(self.data):
                out[k] = self.data[k].consume(tx, tx.params[k][i])
        return outs

    # Definition for the _consume_output method
    def _consume_output(self, tx, output):
        # Return a json object with the returned value from the script's consume_output method
        # Or return blank
        # And return the amount returned from the consume method()
        return {
            "script": self.script.consume_output(tx, output.script) or {},
            "amount": self.amount.consume(tx, output.amount)
        }

    # Definition for the construct method
    def construct(self, tx, outputs):
        # Outputs should be a list
        assert type(outputs) == list, "outputs should be a list"
        # Check that the number of outputs is within the acceptable range
        assert self.min <= len(outputs) <= self.max

        # Iterate through this class instance's data member variable's items
        # Check that there's no namespace conflict
        # Construct the outputs from the items and append them to an l object
        # I'm not sure where the l object goes
        for (k, t) in self.data.items():
            assert k not in tx.params, ('Namespace conflict on "%s"' % k)
            l = tx.params[k] = []
            for out in outputs:
                p = t.construct(tx, out[k])
                l.append(p)

        # Iterate through all the outputs, construct a transaction's outputs for each
        return [self._construct_output(tx, out) for out in outputs]

    # Definition of the _construct_output() method
    def _construct_output(self, tx, spec):
        # Not sure if TxOut is a function or something declared here?
        # Anyhow, return the amount and script, after acquiring them
        return TxOut(
            amount = self.amount.construct(tx, spec.get('amount')),
            script = self.script.construct_output(tx, spec.get('script', {}))
        )


# An option output, provided via script
def OptionalOutput(script):
    # TODO: this should maybe be a class so it can have nice error messages an such
    return Outputs(script, min=0, max=1)


# Output class definition
class Output(Outputs):
    # Accepts some args and "kwargs"? kw arguments?
    def __init__(self, *args, **kwargs):
        # dict() creates a dictionary
        # update() inserts the specificed items into the kwargs' dictionary
        # These are pointers, so presumably this is going straight into the provided object's dictionary
        kwargs.update(dict(min=1, max=1))
        # The super() method here returns methods from this specific object, allowing them to be called from outside this constructor definition
        super(Output, self).__init__(*args, **kwargs)

    # Definition for the construct() method
    def construct(self, tx, output):
        # Once again, we're returning the ability to perform this function outside of itself
        return super(Output, self).construct(tx, [output])

    # Definition for the consume() method
    def consume(self, tx, output):
        # Once again, super() is returning the ability to call this function outside of itself
        return super(Output, self).consume(tx, output)[0]



# Definition for the P2PKH class
# P2PKH is a common type of transaction in Bitcoin
class P2PKH:
    # Definition for the consume_input() method of the P2PKH class
    def consume_input(self, tx, inp):
        # Return the parsed transaction
        return inp.script.parse_p2pkh()

    # Definition for the consume_output() method of the P2PKH class
    def consume_output(self, tx, script):
        # Parse from the provided script
        return script.parse_p2pkh()

    # Construction_input() method
    def construct_input(self, tx, spec):
        # Acquire the from_address
        return ScriptSig.from_address(spec['address'])

    # Construction_output() method
    def construct_output(self, tx, spec):
        # Acquire the from_address
        return ScriptPubKey.from_address(spec['address'])


class SpendBy:
    """
    SpendBy ensures that an output is spent by a given type of input

    SpendBy make either use a dynamic or fixed pubkey.
    If it's fixed (provided in constructor), it does not expect to find
    it in tx spec and does not provide it in validated spec.

    """
    # Constructor
    # By default, no fixed pubkey is required
    def __init__(self, name, pubkey=None):
        self.name = name
        self.pubkey = pubkey

        # TODO: sanity check on structure? make sure that inputs and outputs are compatible
    
    # Consume the output
    def consume_output(self, tx, script):
        # When checking the output there's nothing to check except the script
        # There's some things here about checking a condition -- so this is where the CC type transaction is called
        return self._check_cond(tx, script.parse_condition())

    # Consume the input
    def consume_input(self, tx, inp):
        # Check input script
        r = self._check_cond(tx, inp.script.parse_condition())

        # Check output of parent tx to make sure link is correct
        p = inp.previous_output

        # Set the input_tx member variable according to the TxValidator() method
        # Set the out_model by calling the get_group... method
        input_tx = TxValidator(tx.app, tx.app.chain.get_tx_confirmed(p[0]))
        out_model = input_tx.get_group_for_output(p[1])
        # Check that the out_model.script call works
        assert self._eq(out_model.script)

        # Return the _check_cond() result
        return r

    # Construct an output, use from_condition()
    def construct_output(self, tx, spec):
        return ScriptPubKey.from_condition(self._construct_cond(tx, spec))

    # Construct an input, use from_condition()
    def construct_input(self, tx, spec):
        return ScriptSig.from_condition(self._construct_cond(tx, spec))

    # Set the name and pubkey of the other to the name and pubkey of this SpendBy class instance 
    def _eq(self, other):
        # Should compare the pubkey here? maybe it's not neccesary
        return (type(self) == type(other) and 
                self.name == other.name and
                self.pubkey == other.pubkey)

    # Definition of the _check_cond() method 
    def _check_cond(self, tx, cond):
        # Either take this instance's pubkey, or if none is available, grab the last from the transaction's stack
        pubkey = self.pubkey or tx.stack.pop()
        # There seems to be some kind of evaluation going on here
        # This may be where the full CC is evaluated?
        c = cc_threshold(2, [mk_cc_eval(tx.app.eval_code), cc_secp256k1(pubkey)])
        # Check that the condition being run is the same as the one provided in the function params
        assert c.is_same_condition(cond)
        # If there's already a self.pubkey, return nothing, otherwise, add the provided pubkey
        return {} if self.pubkey else { "pubkey": pubkey }

    # Definition of the _construct_cond() method
    def _construct_cond(self, tx, script_spec):
        # Set the provided self-s pubkey to a variable called pubkey
        pubkey = self.pubkey
        # Error checking
        if pubkey:
            assert not script_spec.get('pubkey'), "pubkey must not be in both spec and schema"
        # Add pubkey to the script_spec
        # Add the pubkey to the end of the transaction's stack
        else:
            pubkey = script_spec['pubkey']
            tx.stack.append(pubkey)
        # Again, I think we're doing an important evaluation here
        return cc_threshold(2, [mk_cc_eval(tx.app.eval_code), cc_secp256k1(pubkey)])


# The Amount class
class Amount():
    # Constructor
    # Default minimum value is 0
    def __init__(self, min=0):
        self.min = min

    # Definition of the consume() function
    def consume(self, tx, amount):
        # Make sure the amount type is an int
        assert type(amount) == int
        # Make sure the amount is greater than the minimum amount
        assert amount >= self.min
        # Return the amount
        return amount

    # Definition of the construct() function
    def construct(self, tx, amount):
        assert type(amount) == int
        # As above, make sure amount is above the minimum amount
        assert amount >= self.min
        return amount


# Definition of the ExactAmount() class
class ExactAmount:
    # Constructor
    def __init__(self, amount):
        self.amount = amount
    
    # Check that the amount requested is equalt to the amount in this instance of the ExactAmount class
    def consume(self, tx, amount):
        assert amount == self.amount

    # Construct an exact amount
    # Make sure amount input is None, then return the amount in this instance
    def construct(self, tx, amount):
        assert amount is None
        return self.amount


# Definition of the RelativeAmount() class
class RelativeAmount:
    # Constructor
    def __init__(self, input_idx, diff=0):
        self.input_idx = input_idx
        # diff is 0 by default
        self.diff = diff

    # Seems to be using the default __sub__() function?
    # Create the relative amount from the contained input_idx and the contained diff - the provided n
    def __sub__(self, n):
        return RelativeAmount(self.input_idx, self.diff - n)

    # Definition of the consume aspect of the RelativeAmount() class
    # Seems to be only validating?
    def consume(self, tx, amount):
        # Grab the self.diff value
        total = self.diff
        # Aggregate the total amount across all of the inputs in the tx provided (?)
        for inp in tx.get_input_group(self.input_idx):
            p = inp.previous_output
            input_tx = tx.app.chain.get_tx_confirmed(p[0])
            total += input_tx.outputs[p[1]].amount

        # Make sure each one is the same as expected
        assert total == amount, "TODO: nice error message"
        # Return the amount
        return amount

    # Definition of the construct() method
    def construct(self, tx, spec):
        # Make sure the provided spec is empty
        assert spec == None, "amount should not be provided for RelativeAmount"

        # Grab the diff value in the provided self
        r = self.diff

        # For each input in the list
        # Something about adding up all the amounts that are confirmed
        for inp in as_list(tx.inputs[self.input_idx]):
            p = inp['previous_output']
            input_tx = tx.app.chain.get_tx_confirmed(p[0])
            r += input_tx.outputs[p[1]].amount

        # Make sure the class instance has requisite balance
        assert r >= 0, "cannot construct RelativeInput: low balance"
        return r

# Either return val as a list, otherwise, (return val as an array?)
def as_list(val):
    return val if type(val) == list else [val]
