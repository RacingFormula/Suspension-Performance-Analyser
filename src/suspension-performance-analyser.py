import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lti, bode, impulse

class SuspensionPerformanceAnalyser:
    def __init__(self, config):
        self.mass = config.get("mass", 300)  # kg, sprung mass
        self.spring_rate = config.get("spring_rate", 15000)  # N/m
        self.damping_coefficient = config.get("damping_coefficient", 1200)  # Ns/m
        self.track_irregularities = config.get("track_irregularities", 0.01)  # m
        self.frequency_range = config.get("frequency_range", (0.1, 20))  # Hz
        self.sample_points = config.get("sample_points", 1000)

    def compute_transfer_function(self):
        # Calculate suspension transfer function
        k = self.spring_rate  # Spring stiffness
        c = self.damping_coefficient  # Damping coefficient
        m = self.mass  # Sprung mass

        # Transfer function: H(s) = 1 / (m*s^2 + c*s + k)
        num = [1]
        den = [m, c, k]
        system = lti(num, den)
        return system

    def frequency_response(self, system):
        # Compute the frequency response of the suspension system
        w = np.logspace(np.log10(self.frequency_range[0]), np.log10(self.frequency_range[1]), self.sample_points)
        _, mag, phase = bode(system, w=w)
        return w, mag, phase

    def impulse_response(self, system):
        # Compute the impulse response of the suspension system
        t, response = impulse(system)
        return t, response

    def run_simulation(self):
        system = self.compute_transfer_function()
        w, mag, phase = self.frequency_response(system)
        t, response = self.impulse_response(system)

        return {
            "frequency": w,
            "magnitude": mag,
            "phase": phase,
            "time": t,
            "impulse_response": response
        }

    def plot_results(self, results):
        # Plot frequency response
        plt.figure(figsize=(14, 8))

        plt.subplot(2, 1, 1)
        plt.semilogx(results["frequency"], results["magnitude"], label="Magnitude")
        plt.title("Frequency Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.semilogx(results["frequency"], results["phase"], label="Phase", color="red")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (degrees)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Plot impulse response
        plt.figure(figsize=(8, 6))
        plt.plot(results["time"], results["impulse_response"], label="Impulse Response", color="green")
        plt.title("Impulse Response")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    config = {
        "mass": 350,
        "spring_rate": 16000,
        "damping_coefficient": 1500,
        "track_irregularities": 0.02,
        "frequency_range": (0.1, 20),
        "sample_points": 1000
    }

    analyser = SuspensionPerformanceAnalyser(config)
    results = analyser.run_simulation()

    print("Simulation Completed.")
    analyser.plot_results(results)