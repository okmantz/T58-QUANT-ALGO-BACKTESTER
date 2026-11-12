from app.search_stacks import STACK_ROLES, certification_door, may_certify


def test_exactly_one_stack_may_certify():
    assert [k for k, v in STACK_ROLES.items() if v["may_certify"]] == ["C"]
    assert certification_door() == "app.discovery_engine" and not may_certify("A") and not may_certify("B")
