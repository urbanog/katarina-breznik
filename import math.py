import random
import numpy as np

import matplotlib.pyplot as plt

# Parameters
N = 17  # Number of segments
height_levels = 19
g = 1.0  # Gravitational constant
k = 1.0  # Elastic constant
T = 1.0  # Temperature

# Function to calculate the energy of the chain
def calculate_energy(chain):
    energy = 0.0
    # Gravitational energy
    for i in range(N):
        energy += g * chain[i]
    # Elastic energy
    for i in range(N - 1):
        energy += k * (chain[i+1] - chain[i])**2
    return energy

# Function to perform a Metropolis step
def metropolis_step(chain, T):
    # Choose a random segment (excluding the fixed ends)
    segment = random.randint(1, N - 2)
    #print(f"Segment {segment=}")

    # Store the old height
    old_height = chain[segment]
    #print(f"{old_height=}")

    # Propose a new height (one unit up or down)
    possible_new_heights = []
    if old_height > 0:
      possible_new_heights.append(old_height - 1)
    if old_height < height_levels - 1:
      possible_new_heights.append(old_height + 1)
    new_height = random.choice(possible_new_heights)
    #print(f"{possible_new_heights=}")
    #print(f"{new_height=}")

    # Create a new chain with the proposed height
    new_chain = chain[:]
    new_chain[segment] = new_height

    # Calculate the energy change
    old_energy = calculate_energy(chain)
    new_energy = calculate_energy(new_chain)
    delta_E = new_energy - old_energy
    #print(f"{delta_E=}")

    # Metropolis acceptance criterion
    if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
        # Accept the new state
        return new_chain
    else:
        # Reject the new state
        return chain

# Initialize the chain with random heights (fixed ends)
def initialize_chain():
    chain = [0] * N
    for i in range(1, N - 1):
        chain[i] = random.randint(0, height_levels - 1)
    return chain

# Run the Metropolis simulation
def run_simulation(T, steps):
    chain = initialize_chain()
    energies = []
    for _ in range(steps):
        chain = metropolis_step(chain, T)
        energies.append(calculate_energy(chain))
    return chain, energies

# Example usage
if __name__ == "__main__":
    # Parameters
    T = 1.0  # Temperature
    steps = 10000  # Number of Metropolis steps

    # Run the simulation
    final_chain, energies = run_simulation(T, steps)

    # Print the final chain configuration
    print("Final chain configuration:", final_chain)

    # Plot the energy history
    plt.plot(energies)
    plt.xlabel("Metropolis step")
    plt.ylabel("Energy")
    plt.title("Energy history")
    plt.show()