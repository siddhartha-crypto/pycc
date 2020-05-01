
# Import everything in the PyCC module
from pycc import *


# Sets the global address, just as with the old CC prototypes
global_addr = {
  "wif": "UsNHJsn9Axq63PwKoUM84RuUByjs83gCWrCixpQ7FGb8ifVQs58a",
  "addr": "RNZgNenJMp9UdxR6exxogjEzqqUT5F7hXC",
  "pubkey": "03997fec500b2405c234724269a59afa0750c3ce10b9240a74deb48b3a852d8b41"
}

# Not sure what a "schema_link" is, but this would seem to be the entry point for the CC module and its transactions
schema_link = SpendBy("faucet.drip", pubkey=global_addr['pubkey'])


# The schema
# The first object appears to contain only the name, faucet. This probably lines up with some of the [i][k] stuff in the pycc.py file
# Then there are two types of transactions, create and drip
# Within these two types, there are inputs and outputs
schema = {
    "faucet": {
        "create": {
            "inputs": [
                Input(P2PKH())
            ],
            "outputs": [
                Output(schema_link)
            ],
        },
        "drip": {
            "inputs": [
                Input(schema_link)
            ],
            "outputs": [
                Output(schema_link, RelativeAmount(0) - 1000),
                Output(P2PKH())
            ]
        },
    }
}


# The evaluation function for this cc, 
def cc_eval(chain, tx_bin, nIn, eval_code):
    return CCApp(schema, eval_code, chain).cc_eval(tx_bin)

# To run komodod with this module:
# > PYTHONPATH=. /path/to/komodod -ac_pycc=pycc.examples.faucet
