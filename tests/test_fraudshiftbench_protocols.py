from fraudshiftbench.protocols import ProtocolContract, default_protocol_contracts


def test_protocol_contract_round_trip() -> None:
    contract = default_protocol_contracts()[0]
    assert ProtocolContract.from_dict(contract.to_dict()) == contract
    assert contract.name
    assert contract.prohibited_claims
