import sumolib
import numpy as np
import matplotlib.pyplot as plt

class SumoVisualizer:
    def __init__(self, sumo_net_file, sumo_route_file):
        self.net = sumolib.net.readNet(sumo_net_file)
        self.route = sumolib.net.readRoutes(sumo_route_file)
        self.sumo_sim = sumolib.simulation.Simulation(self.net, self.route)

    def visualize_step(self, agent_state, agent_action):
        # Update the SUMO simulation with the agent's action.
        self.sumo_sim.step(agent_action)

        # Get the current state of the SUMO simulation.
        sim_state = self.sumo_sim.get_state()

        # Visualize the SUMO simulation state.
        plt.figure()
        plt.imshow(sim_state.vehicles)
        plt.colorbar()
        plt.title("SUMO Simulation State")
        plt.show()

    def close(self):
        self.sumo_sim.close()