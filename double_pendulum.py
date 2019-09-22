import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DoublePendulum():
    def __init__(self, M_1 = 1, L_1 = 1, M_2 = 1, L_2 = 1, g = 9.81):
        self.M_1 = M_1
        self.L_1 = L_1
        self.M_2 = M_2
        self.L_2 = L_2
        self.g = g

    """
    Takes in u and stores its value in four new variables, theta1, omega1, theta2, omega2.
    Then calculates domega1 and domega2 from the formula provided. Returns result.
    """
    def __call__(self, t, u):
        theta1, omega1, theta2, omega2 = u
        dtheta = theta2 - theta1

        domega1 = (self.M_2 * self.L_1 * omega1 ** 2 * np.sin(dtheta) * np.cos(dtheta) + self.M_2 * self.g * np.sin(
            theta2) * np.cos(dtheta) + self.M_2 * self.L_2 * omega2 ** 2 * np.sin(dtheta) - (
                               self.M_1 + self.M_2) * self.g * np.sin(theta1)) / (
                              (self.M_1 + self.M_2) * self.L_1 - self.M_2 * self.L_1 * (np.cos(dtheta)) ** 2)
        domega2 = (-self.M_2 * self.L_2 * omega2 ** 2 * np.sin(dtheta) * np.cos(dtheta) + (
                    self.M_1 + self.M_2) * self.g * np.sin(theta1) * np.cos(dtheta) - (
                               self.M_1 + self.M_2) * self.L_1 * omega1 ** 2 * np.sin(dtheta) - (
                               self.M_1 + self.M_2) * self.g * np.sin(theta2)) / (
                              (self.M_1 + self.M_2) * self.L_2 - self.M_2 * self.L_2 * (np.cos(dtheta)) ** 2)

        dtheta1 = omega1
        dtheta2 = omega2

        return dtheta1, domega1, dtheta2, domega2

    """
    Solve solves the ODE we have and returns the value in four private variables.
    Also returns radians if the input is in degrees.
    """
    def solve(self, y0, T, dt, angles):
        if angles == "deg":
            dtheta1[0] = np.pi/180*dtheta[0]
            domega1[0] = np.pi/180*domega1[0]
            dtheta2[0] = np.pi/180*dtheta2[0]
            domega2[0] = np.pi/180*domega[0]
        n = np.linspace(0, T, int(T/dt))
        sol = scipy.integrate.solve_ivp(self, t_span=(0, T), y0=y0, t_eval=n, method = "Radau")
        self._t = sol.t
        self._theta1, self._omega1, self._theta2, self._omega2 = sol.y

        self.dt = dt

    """
    Adding properties.
    The first three properties will raise an attribute error if called upon before solve is called
    as they will not be "created" before solve is colled.
    """

    @property
    def t(self):
        try:
            return self._t
        except AttributeError:
            raise AttributeError("Solve has not been called")

    @property
    def theta1(self):
        try:
            return self._theta1
        except AttributeError:
            raise AttributeError("Solve has not been called")

    @property
    def theta2(self):
        try:
            return self._theta2
        except AttributeError:
            raise AttributeError("Solve has not been called")

    @property
    def x1(self):
        return self.L_1*np.sin(self._theta1)

    @property
    def y1(self):
        return -self.L_1*np.cos(self._theta1)

    @property
    def x2(self):
        return self.x1 + self.L_2*np.sin(self._theta2)

    @property
    def y2(self):
        return self.y1 - self.L_2*np.cos(self._theta2)

    @property
    def potential(self):
        return (self.M_1*self.g*(self.y1 + self.L_1) + self.M_2*self.g*(self.y2 + self.L_1 + self.L_2))

    @property
    def vx1(self):
        return np.gradient(self.x1, self._t)

    @property
    def vy1(self):
        return np.gradient(self.y1, self._t)

    @property
    def vx2(self):
        return np.gradient(self.x2, self._t)

    @property
    def vy2(self):
        return np.gradient(self.y2, self._t)

    @property
    def kinetic(self):
        return ((1/2)*self.M_1*(self.vx1**2 + self.vy1**2) + (1/2)*self.M_2*(self.vx2**2 + self.vy2**2))

    """
    Functions that creates the figure the animation will play in, a function for showing the animation
    and a function for saving the animation. The save function will only work if you have
    the correct packages installed. 
    """

    def create_animation(self):
        fig = plt.figure()

        plt.axis('equal')
        plt.axis('off')
        plt.axis((-3, 3, -3, 3))

        self.pendulums, = plt.plot([], [], 'o-', lw = 2)

        self.animation = animation.FuncAnimation(fig, self._next_frame, frames=range(len(self.x1)), repeat=None, interval=1000*self.dt, blit=True)

    def _next_frame(self, i):
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.y1[i], self.y2[i]))
        return self.pendulums,

    def show_animation(self):
        plt.show()

    def save_animation(self, name):
        self.animation.save(name, fps=60)

if __name__ == "__main__":
    Double_pendulum1 = DoublePendulum()
    Double_pendulum1.solve((1, 10, 4, 13), 10, 1/100, "rad")
    Double_pendulum1.create_animation()
    Double_pendulum1.show_animation()
    Double_pendulum1.save_animation("pendulum_motion.mp4")


    plt.title("Double pendulum over time")
    plt.plot(Double_pendulum1.t, Double_pendulum1.theta2, label = "theta2")
    plt.plot(Double_pendulum1.t, Double_pendulum1.theta1, label = "theta1")
    plt.show()
    plt.title("Kinetic, Potential and total energy of the pendulum")
    plt.plot(Double_pendulum1.t, Double_pendulum1.potential, label = "Potential energy")
    plt.plot(Double_pendulum1.t, Double_pendulum1.kinetic, label = "Kinetic energy")
    plt.plot(Double_pendulum1.t, Double_pendulum1.potential + Double_pendulum1.kinetic, label = "Total energy")
    plt.legend()
    plt.show()
