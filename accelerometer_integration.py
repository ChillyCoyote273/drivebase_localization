import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd


def normalize_angle(theta: float) -> float:
    theta %= 2 * np.pi
    if theta >= np.pi:
        theta -= 2 * np.pi
    return theta


class Pose:
    def __init__(self, t: float, x: float, y: float, theta: float, dx: float, dy: float, dtheta: float):
        self.t = t
        self.x = x
        self.y = y
        self.theta = normalize_angle(theta)
        self.dx = dx
        self.dy = dy
        self.dtheta = dtheta
    
    def __str__(self):
        return f'Pose {"{"}\n\tt:\t{self.t}\n\tx:\t{self.x}\n\ty:\t{self.y}\n\ttheta:\t{self.theta}\n\tdx:\t{self.dx}\n\tdy:\t{self.dy}\n\tdtheta:\t{self.dtheta}\n{"}"}' # dw about it
    
    def __repr__(self):
        return self.__str__()
    
    def copy(self):
        return Pose(self.t, self.x, self.y, self.theta, self.dx, self.dy, self.dtheta)
    
    def to_list(self) -> list[float]:
        return [self.t, self.x, self.y, self.theta, self.dx, self.dy, self.dtheta]


def pose_exponential(current_pose: Pose, dt: float, dx: float, dy: float, dtheta: float) -> Pose:
    dtheta = normalize_angle(dtheta)
    delta_vector = np.array([dx, dy, dtheta])
    theta = current_pose.theta
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    curvature_matrix = np.array([
        [np.sin(dtheta) / dtheta, (np.cos(dtheta) - 1) / dtheta, 0],
        [(1 - np.cos(dtheta)) / dtheta, np.sin(dtheta) / dtheta, 0],
        [0, 0, 1]
    ]) if dtheta != 0 else np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    delta_vector = rotation_matrix @ curvature_matrix @ delta_vector
    x, y, theta = current_pose.x + delta_vector[0], current_pose.y + delta_vector[1], current_pose.theta + delta_vector[2]
    velocity = np.array([dx, dy, dtheta]) / dt
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    velocity = rotation_matrix @ velocity
    dx, dy, dtheta = velocity[0], velocity[1], velocity[2]
    t = current_pose.t + dt
    return Pose(t, x, y, theta, dx, dy, dtheta)


def calculate_path(times: np.ndarray, xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray) -> list[Pose]:
    dts = np.diff(times)
    dxs = np.diff(xs)
    dyx = np.diff(ys)
    dthetas = np.diff(thetas)

    current_pose = Pose(0, 0, 0, 0, 0, 0, 0)
    poses = [current_pose]

    for dt, dx, dy, dtheta in zip(dts, dxs, dyx, dthetas):
        current_pose = pose_exponential(current_pose, dt, dx, dy, dtheta)
        poses.append(current_pose)
    
    return poses


def path_to_dataframe(path: list[Pose]) -> pd.DataFrame:
    return pd.DataFrame([pose.to_list() for pose in path], columns=['time', 'x', 'y', 'theta', 'x_vel', 'y_vel', 'omega'])


def load_data() -> pd.DataFrame:
    return pd.read_csv('data/IMUvsRealData.csv')


def display_path(path: pd.DataFrame) -> None:
    _, ((xy, ty), (xt, _)) = plt.subplots(2, 2, sharex='col', sharey='row')
    xy.plot(path['x'], path['y'])
    xt.plot(path['x'], path['time'])
    ty.plot(path['time'], path['y'])
    _, ((xyv, tyv), (xtv, _)) = plt.subplots(2, 2, sharex='col', sharey='row')
    xyv.plot(path['x_vel'], path['y_vel'])
    xtv.plot(path['x_vel'], path['time'])
    tyv.plot(path['time'], path['y_vel'])
    plt.show()


def main():
    raw_data = load_data()
    times = raw_data['time'].to_numpy()
    accels = raw_data['accel_x'].to_numpy()

    delta_ts = np.diff(times) * 1000
    print(delta_ts.mean())
    print(delta_ts.std())

    plt.hist(delta_ts, 15)
    plt.show()


if __name__ == '__main__':
    main()
