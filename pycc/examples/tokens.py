
from pycc import *


# Definition of the Token class
class Token:
    # Definition of the consume() function
    def consume(self, tx, spec):
        tot_input = tot_output = 0

        # Grab all the transactions that are confirmed
        for inp in tx.get_input_group(0):
            input_tx = TxValidator(tx.app, tx.app.chain.get_tx_confirmed(inp.previous_output[0]))
            # This will break because the output index needs to be translated for the group
            # In reality the thing to do is to call validate() to get the spec, but not do I/O
            # TODO: check token ID on the input
            # TODO: check that input tx has right eval code
            tot_input += input_tx.params['tokenoshi'][inp.previous_output[1]]

        # Iterate through all the outputs and add them to the total output
        for out in spec['outputs'][0]:
            tot_output += out['tokenoshi']

        # Make sure the total value of the inputs are greater than or equal to the total value of the outputs
        assert tot_input >= tot_output
        # return token ID

    # Construct definition
    def construct(self, tx, token_id):
        tx.params['token'] = token_id


# Another token_link, need to ask about these
token_link = SpendBy('token.transfer')


# The outputs for the Tokens module
outputs = [
    Outputs(
        # Does the SpendBy() thing above
        script = token_link,
        # Run the ExactAmount() function
        amount = ExactAmount(0),
        # This must state something about how many tokens exist in the contract instance
        data = {"tokenoshi": Amount(min=1)}
    ),
    # Keep an optional output here with the P2PKH()
    OptionalOutput(P2PKH())
]


# The schema for the Tokens module
schema = {
    # The name is token
    "token": {
        # There are create and transfer objects in the schema, each with inputs and outputs
        # The inputs requires P2PKH()
        # The transfer allows for both P2PKH() and token_link
        "create": {
            "inputs": [
                Input(P2PKH())
            ],
            "outputs": outputs
        },
        "transfer": {
            "inputs": [
                Inputs(token_link),
                Input(P2PKH())
            ],
            "outputs": outputs,
            "token": Token
        },
    }
}

