import os
import pickle
import numpy as np
from absl import flags, app
import meshio

flags.DEFINE_string("rollout_path", None, help="Path to rollout pickle file")

FLAGS = flags.FLAGS


def main(unused):
    if not FLAGS.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")

    for j in range(0, 36):
        with open(FLAGS.rollout_path + '\\rollout_test_' + str(j) + '.pkl', "rb") as file:
            rollout_data = pickle.load(file)

        trajectory = np.concatenate([rollout_data["initial_velocity"], rollout_data['predicted_rollout']], axis=0)
        realtrajectory = np.concatenate([rollout_data["initial_velocity"], rollout_data['ground_truth_rollout']],
                                        axis=0)

        nodes = rollout_data['locations']
        nodes_x = rollout_data['locations'][:, 0]
        nodes_y = rollout_data['locations'][:, 1]
        edges = rollout_data['connections']

        cells = [('triangle', edges)]

        if not os.path.exists('D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j)):
            os.makedirs('D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j))

        mean_mse = []
        mean_x_mse = []
        mean_y_mse = []

        for i in range(0, len(trajectory)):
            t_step = trajectory[i]
            t_x = t_step[:, 0]
            t_y = t_step[:, 1]
            t = np.sqrt(np.add(np.square(t_x), np.square(t_y)))

            if i == 199:
                mesh = meshio.Mesh(nodes, cells,
                                   point_data={'Velocity_Magnitude': t, 'X-Velocity': t_x, 'Y-Velocity': t_y})
                mesh.write(
                    'D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j) + '/frame' + str(i) + '.vtk')
            else:
                rt_step = realtrajectory[i]
                rt_x = rt_step[:, 0]
                rt_y = rt_step[:, 1]
                rt = np.sqrt(np.add(np.square(rt_x), np.square(rt_y)))

                mse = np.square(np.subtract(t, rt))
                avg_mse = np.mean(mse)
                mean_mse = np.append(mean_mse, avg_mse)

                x_mse = np.square(np.subtract(t_x, rt_x))
                avg_x_mse = np.mean(x_mse)
                mean_x_mse = np.append(mean_x_mse, avg_x_mse)

                y_mse = np.square(np.subtract(t_y, rt_y))
                avg_y_mse = np.mean(y_mse)
                mean_y_mse = np.append(mean_y_mse, avg_y_mse)

                mesh = meshio.Mesh(nodes, cells,
                                   point_data={'Velocity_Magnitude': t, 'X-Velocity': t_x, 'Y-Velocity': t_y,
                                               'Ground': rt, 'MSE': mse})
                mesh.write(
                    'D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j) + '/frame' + str(i) + '.vtk')

        np.savetxt('D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j) + '/mean_mse.csv', mean_mse, delimiter=",")
        np.savetxt('D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j) + '/mean_x_mse.csv', mean_x_mse, delimiter=",")
        np.savetxt('D:/Users/abdel/Documents/DISSO_FILES/experimental_render/Rollout' + str(j) + '/mean_y_mse.csv', mean_y_mse, delimiter=",")


if __name__ == "__main__":
    app.run(main)
