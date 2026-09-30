import json
from pathlib import Path
def test_week1_protocol_locked():
    config_path = Path("configs/week1_protocol.json")
    assert config_path.exists(), "week1_protocol.json missing"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    
    assert config["training_protocol_status"] == "locked", "Protocol must be locked"
    protocol = config["internal_ablation_protocol"]
    assert protocol["max_epochs"] == 100, "Epoch budget must be 100"
    assert "max_epochs_candidate" not in protocol, "Candidate keys must be removed"
    assert protocol["batch_size"] == 32
    print("test_week1_protocol_locked PASS")


if __name__ == "__main__":
    test_week1_protocol_locked()

