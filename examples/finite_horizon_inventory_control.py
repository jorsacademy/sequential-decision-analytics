import numpy as np
import matplotlib.pyplot as plt


class FiniteHorizonInventoryModel:
    """Finite-horizon stochastic inventory control solved by backward induction."""

    def __init__(
        self,
        capacity=10,
        horizon=8,
        demand_values=(0, 1, 2, 3, 4),
        demand_probabilities=(0.10, 0.20, 0.35, 0.25, 0.10),
        selling_price=8.0,
        order_cost=3.0,
        holding_cost=1.0,
        stockout_penalty=2.0,
        salvage_value=0.0,
    ):
        if capacity < 0 or horizon <= 0:
            raise ValueError("capacity must be non-negative and horizon must be positive.")

        self.capacity = int(capacity)
        self.horizon = int(horizon)
        self.demand_values = np.asarray(demand_values, dtype=int)
        self.demand_probabilities = np.asarray(demand_probabilities, dtype=float)
        self.selling_price = float(selling_price)
        self.order_cost = float(order_cost)
        self.holding_cost = float(holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.salvage_value = float(salvage_value)

        if len(self.demand_values) != len(self.demand_probabilities):
            raise ValueError("Demand values and probabilities must have the same length.")
        if np.any(self.demand_values < 0):
            raise ValueError("Demand values must be non-negative.")
        if np.any(self.demand_probabilities < 0):
            raise ValueError("Demand probabilities must be non-negative.")
        if not np.isclose(self.demand_probabilities.sum(), 1.0):
            raise ValueError("Demand probabilities must sum to 1.")

    def one_period_return(self, inventory, order_quantity, demand):
        """Return immediate profit and next inventory for one demand outcome."""
        available = inventory + order_quantity
        sales = min(available, demand)
        lost_sales = max(demand - available, 0)
        ending_inventory = max(available - demand, 0)

        profit = (
            self.selling_price * sales
            - self.order_cost * order_quantity
            - self.holding_cost * ending_inventory
            - self.stockout_penalty * lost_sales
        )
        return profit, ending_inventory

    def solve(self):
        """Compute an optimal finite-horizon policy by backward induction."""
        values = np.zeros((self.horizon + 1, self.capacity + 1), dtype=float)
        policy = np.zeros((self.horizon, self.capacity + 1), dtype=int)

        inventories = np.arange(self.capacity + 1)
        values[self.horizon] = self.salvage_value * inventories

        for t in range(self.horizon - 1, -1, -1):
            for inventory in inventories:
                feasible_orders = range(self.capacity - inventory + 1)
                action_values = []

                for order_quantity in feasible_orders:
                    expected_value = 0.0

                    for demand, probability in zip(
                        self.demand_values,
                        self.demand_probabilities,
                    ):
                        immediate_profit, next_inventory = self.one_period_return(
                            inventory,
                            order_quantity,
                            int(demand),
                        )
                        expected_value += probability * (
                            immediate_profit + values[t + 1, next_inventory]
                        )

                    action_values.append(expected_value)

                best_order = int(np.argmax(action_values))
                policy[t, inventory] = best_order
                values[t, inventory] = action_values[best_order]

        return values, policy

    def simulate_policy(self, policy, initial_inventory=0, seed=42):
        """Simulate one sample path under a supplied policy."""
        if not 0 <= initial_inventory <= self.capacity:
            raise ValueError("initial_inventory must be between 0 and capacity.")

        rng = np.random.default_rng(seed)
        inventory = int(initial_inventory)
        total_profit = 0.0
        history = []

        for t in range(self.horizon):
            order_quantity = int(policy[t, inventory])
            demand = int(
                rng.choice(self.demand_values, p=self.demand_probabilities)
            )
            profit, next_inventory = self.one_period_return(
                inventory,
                order_quantity,
                demand,
            )

            history.append(
                (t + 1, inventory, order_quantity, demand, next_inventory, profit)
            )
            total_profit += profit
            inventory = next_inventory

        total_profit += self.salvage_value * inventory
        return history, total_profit


def plot_policy(policy):
    """Visualize the optimal order quantity by time and inventory state."""
    plt.figure(figsize=(10, 5))
    image = plt.imshow(policy, aspect="auto", origin="upper")
    plt.colorbar(image, label="Optimal Order Quantity")
    plt.title("Finite-Horizon Optimal Inventory Policy")
    plt.xlabel("Inventory State")
    plt.ylabel("Decision Period")
    plt.yticks(np.arange(policy.shape[0]), np.arange(1, policy.shape[0] + 1))
    plt.xticks(np.arange(policy.shape[1]))
    plt.tight_layout()
    plt.show()


def main():
    model = FiniteHorizonInventoryModel()
    values, policy = model.solve()

    print("Optimal order quantities by period and inventory state:")
    print(policy)
    print(f"Expected optimal value from zero inventory: {values[0, 0]:.2f}")

    history, total_profit = model.simulate_policy(
        policy,
        initial_inventory=0,
        seed=42,
    )

    print("\nSample path:")
    print("Period | Start Inv. | Order | Demand | End Inv. | Profit")
    for row in history:
        period, start_inv, order, demand, end_inv, profit = row
        print(
            f"{period:>6} | {start_inv:>10} | {order:>5} | "
            f"{demand:>6} | {end_inv:>8} | {profit:>6.2f}"
        )

    print(f"\nSample-path total profit: {total_profit:.2f}")
    plot_policy(policy)


if __name__ == "__main__":
    main()
