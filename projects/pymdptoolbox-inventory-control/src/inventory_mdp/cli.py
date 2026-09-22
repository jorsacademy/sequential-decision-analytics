"""Command-line demo for the inventory MDP."""

from .model import InventoryConfig, solve_inventory_mdp


def main() -> None:
    config = InventoryConfig()
    vi_policy, _ = solve_inventory_mdp(config, "value_iteration")
    pi_policy, _ = solve_inventory_mdp(config, "policy_iteration")

    print("state -> value_iteration / policy_iteration")
    for state, (vi_action, pi_action) in enumerate(zip(vi_policy, pi_policy, strict=True)):
        print(f"{state:>2} -> {vi_action:>2} / {pi_action:>2}")


if __name__ == "__main__":
    main()
