# =============================================================================
# © 2025 Lance Thomas Davidson. All rights reserved.
#
# This source code, mathematical framework, recursive Hamiltonian model, and all
# derivative computational formulations herein are the intellectual property of
# Lance Thomas Davidson. No part of this system — whether conceptual, structural,
# mathematical, symbolic, recursive, or computational — may be copied, modified,
# reverse-engineered, incorporated into another system, or used as the basis for
# any derivative work, simulation model, algorithmic framework, or commercial
# platform, whether in whole or in part, without the express written consent of
# Lance Thomas Davidson.
#
# This includes, but is not limited to:
# - Replication of the recursive logarithmic-trigonometric elastic Hamiltonian
# - Adaptation of D-module algebraic twist integrations in tensor systems
# - Any use in quantum simulations, financial trading algorithms, or geometric AI
# - Academic publication of derivative or inferentially similar systems
#
# Violators of this license will be subject to applicable legal enforcement under
# international intellectual property and software copyright law.
#
# Express licensing inquiries must be directed to:
#   Lance Thomas Davidson — [lancedavidson77@gmail.com]
# =============================================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --------------------------- Dissection Hamiltonian (NumPy) ---------------------------

class DissectionHamiltonian:
    def __init__(self, num_pieces=3):
        self.num_pieces = num_pieces
        self.epsilon = 1e-9

    def RigidTransform(self, x, theta):
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        return R @ x

    def Logarithmic_Euclidean(self, x_i, x_j, x_i_prime, x_j_prime):
        norm_original = np.linalg.norm(x_i - x_j) + self.epsilon
        norm_transformed = np.linalg.norm(x_i_prime - x_j_prime) + self.epsilon
        return np.log(norm_original / norm_transformed)

    def Trig_Elasticity(self, theta, dtheta_dt):
        return dtheta_dt * (1 - np.cos(theta)) * np.log(1 + theta**2 + self.epsilon)

    def Curvature_Energy(self, kappa, d2kappa_dt2):
        return (1 / (1 + np.log(1 + kappa**2 + self.epsilon))) * d2kappa_dt2

    def RecursiveTier(self, x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list, depth=3):
        if depth == 0:
            return 0
        sum_terms = 0
        for i in range(self.num_pieces):
            for j in range(i + 1, self.num_pieces):
                x_i, x_j = x_list[i], x_list[j]
                theta_i, theta_j = theta_list[i], theta_list[j]
                x_i_prime = self.RigidTransform(x_i, theta_i)
                x_j_prime = self.RigidTransform(x_j, theta_j)
                log_euclidean = self.Logarithmic_Euclidean(x_i, x_j, x_i_prime, x_j_prime)
                trig_elastic = self.Trig_Elasticity(theta_list[i], dtheta_dt_list[i])
                curvature_energy = self.Curvature_Energy(kappa_list[i], d2kappa_dt2_list[i])
                sum_terms += log_euclidean + trig_elastic + curvature_energy
        return np.log(1 + sum_terms + self.RecursiveTier(
            x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list, depth - 1))

    def EvaluateHamiltonian(self, x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list):
        return self.RecursiveTier(x_list, theta_list, dtheta_dt_list, kappa_list, d2kappa_dt2_list)

# --------------------------- Twisted Hamiltonian (TensorFlow) ---------------------------

class TwistedHamiltonian:
    def __init__(self, num_points=3, twist_lambda=1.0):
        self.num_points = num_points
        self.twist_lambda = tf.constant(twist_lambda, dtype=tf.float64)
        self.epsilon = tf.constant(1e-9, dtype=tf.float64)

    def euclidean_metric(self, x):
        return tf.eye(len(x[0]), dtype=tf.float64)

    def twist_potential(self, x_k):
        return self.twist_lambda * tf.reduce_sum(tf.square(x_k))

    def sin_log_term(self, theta, grad):
        return tf.math.sin(theta) * tf.math.log(1 + self.epsilon + tf.reduce_sum(tf.square(grad)))

    def compute_second_derivatives(self, f, x):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(x)
            with tf.GradientTape() as tape1:
                tape1.watch(x)
                f_val = f(x)
            grad = tape1.gradient(f_val, x)
        hessian_rows = [tape2.gradient(grad[:, i], x) for i in range(grad.shape[1])]
        hessian = tf.stack(hessian_rows, axis=1)
        del tape2
        return grad, hessian

    def recursive_elastic_hamiltonian(self, f, x, theta_list, depth=3):
        if depth == 0:
            return tf.constant(0.0, dtype=tf.float64)
        grad, hess = self.compute_second_derivatives(f, x)
        g_inv = self.euclidean_metric(x)
        kinetic_term = -tf.reduce_sum(tf.multiply(hess, g_inv))
        potential_sum = tf.constant(0.0, dtype=tf.float64)
        for i in range(self.num_points):
            x_k = x[i]
            theta_k = theta_list[i]
            V_lambda = self.twist_potential(x_k)
            trig_term = self.sin_log_term(theta_k, grad[i])
            potential_sum += V_lambda * trig_term
        total = kinetic_term + potential_sum
        recursive_contrib = self.recursive_elastic_hamiltonian(f, x, theta_list, depth - 1)
        return tf.math.log(1 + total + recursive_contrib)

    def EvaluateHamiltonian(self, f, x, theta_list, depth=3):
        return self.recursive_elastic_hamiltonian(f, x, theta_list, depth)

def scalar_field(x):
    return tf.reduce_sum(x ** 2)

# --------------------------- Unified Runtime Execution ---------------------------

if __name__ == "__main__":
    # NumPy system setup
    hamiltonian_numpy = DissectionHamiltonian(num_pieces=3)
    x_list_np = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3)/2])]
    theta_np = [0.0, np.pi/6, -np.pi/6]
    dtheta_dt_np = [0.01, 0.02, 0.015]
    kappa_np = [0.0, 0.0, 0.0]
    d2kappa_dt2_np = [0.001, 0.001, 0.001]
    H_numpy = hamiltonian_numpy.EvaluateHamiltonian(x_list_np, theta_np, dtheta_dt_np, kappa_np, d2kappa_dt2_np)

    # TensorFlow system setup
    x_tf = tf.Variable([[1.0, 0.0], [0.5, 0.866], [0.0, 1.0]], dtype=tf.float64)
    theta_tf = tf.constant([0.1, 0.2, 0.3], dtype=tf.float64)
    hamiltonian_tf = TwistedHamiltonian(num_points=3, twist_lambda=1.5)
    H_tf = hamiltonian_tf.EvaluateHamiltonian(scalar_field, x_tf, theta_tf, depth=3)

    # Final output
    print("\n=== Unified Dissection + Twisted D-Module Hamiltonian Output ===")
    print(f"Dissection Hamiltonian (NumPy): {H_numpy}")
    print(f"Twisted Hamiltonian (TensorFlow): {H_tf.numpy()}")
    print(f"Combined Total Hamiltonian: {H_numpy + H_tf.numpy()}")


# ---------------------- Energy Component Breakdown ----------------------
    print("\n--- Component Breakdown ---")
    print("Rotation angles (theta):", theta_np)
    print("Angular velocity (dtheta/dt):", dtheta_dt_np)
    print("Curvature values (kappa):", kappa_np)
    print("Curvature acceleration (d²kappa/dt²):", d2kappa_dt2_np)

    # Optional: output as LaTeX expression (symbolic report-ready)
    print("\n--- LaTeX Export ---")
    print(f"H_{{\\text{{NumPy}}}} = {H_numpy:.6f}")
    print(f"H_{{\\text{{TensorFlow}}}} = {H_tf.numpy():.6f}")
    print(f"H_{{\\text{{Total}}}} = {H_numpy + H_tf.numpy():.6f}")

    # Optional: plot setup (not dynamic)
    try:
        fig, ax = plt.subplots()
        triangle = np.array(x_list_np + [x_list_np[0]])
        ax.plot(triangle[:, 0], triangle[:, 1], marker='o')
        ax.set_title("Piece Geometry (Dissection Domain)")
        ax.axis('equal')
        plt.show()
    except Exception as e:
        print("Plotting skipped due to error:", e)

    # Optional: export to file
    with open("hamiltonian_output.txt", "w") as f:
        f.write("Dissection + Twisted D-Module Hamiltonian Analysis\n")
        f.write(f"Dissection Hamiltonian: {H_numpy}\n")
        f.write(f"Twisted Hamiltonian: {H_tf.numpy()}\n")
        f.write(f"Total Energy: {H_numpy + H_tf.numpy()}\n")
