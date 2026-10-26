import os
import argparse
import yaml
import numpy as np

from easydict import EasyDict
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

from mid import MID
from dataset import get_timesteps_data
from evaluation.trajectory_utils import prediction_output_to_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MID trajectory predictions (diffusion generate)")
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)  # e.g. eth/
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--num_scenes', type=int, default=1)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='vis')
    parser.add_argument('--node_type', type=str, default='PEDESTRIAN', help='node type string, default PEDESTRIAN')
    parser.add_argument('--num_points', type=int, default=12, help='prediction horizon steps used in generate')
    parser.add_argument('--num_samples', type=int, default=20, help='number of sampled trajectories')
    return parser.parse_args()


def build_agent(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config['config'] = args.config
    config['dataset'] = args.dataset[:-1]  # strip trailing '/'
    config['exp_name'] = os.path.basename(args.config).split('.')[0]

    config['eval_mode'] = True
    config['eval_at'] = args.epoch

    config = EasyDict(config)
    agent = MID(config)
    return agent


def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(predictions.shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=cmap[node.type.value],
                    linewidth=line_width, alpha=line_alpha)

        ax.plot(future[:, 0],
                future[:, 1],
                'w--',
                path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

        circle = plt.Circle((history[-1, 0], history[-1, 1]),
                            node_circle_size,
                            facecolor='g',
                            edgecolor='k',
                            lw=circle_edge_width,
                            zorder=3)
        ax.add_artist(circle)

    ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(
        prediction_output_dict,
        dt,
        max_hl,
        ph,
        map=map
    )

    if len(prediction_dict.keys()) == 0:
        return

    ts_key = list(prediction_dict.keys())[0]
    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)

    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, **kwargs)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    agent = build_agent(args)

    eval_env = agent.eval_env
    scenes = eval_env.scenes[:args.num_scenes]

    max_hl = agent.hyperparams['maximum_history_length']
    ph = agent.hyperparams['prediction_horizon']

    img_count = 0
    for scene_idx, scene in enumerate(scenes):
        print(f"Visualizing scene {scene_idx + 1}/{len(scenes)} with {scene.timesteps} timesteps")

        for t in range(0, scene.timesteps, args.stride):
            timesteps = np.arange(t, t + 10)

            batch = get_timesteps_data(
                env=eval_env,
                scene=scene,
                t=timesteps,
                node_type=args.node_type,
                state=agent.hyperparams['state'],
                pred_state=agent.hyperparams['pred_state'],
                edge_types=eval_env.get_edge_types(),
                min_ht=7,
                max_ht=max_hl,
                min_ft=args.num_points,
                max_ft=args.num_points,
                hyperparams=agent.hyperparams,
            )

            if batch is None:
                continue

            test_batch, nodes, timesteps_o = batch[0], batch[1], batch[2]

            traj_pred = agent.model.generate(
                test_batch,
                args.node_type,
                num_points=args.num_points,
                sample=args.num_samples,
                bestof=True
            )  # expected: (B, S, T, 2)

            predictions_dict = {}
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict:
                    predictions_dict[ts] = {}
                predictions_dict[ts][nodes[i]] = np.transpose(traj_pred[:, [i]], (1, 0, 2, 3))

            fig, ax = plt.subplots(figsize=(6, 6))
            visualize_prediction(
                ax=ax,
                prediction_output_dict=predictions_dict,
                dt=scene.dt,
                max_hl=max_hl,
                ph=ph,
                map=None,
            )
            ax.set_title(f"Scene {scene_idx}, t={t}")
            save_path = os.path.join(args.output_dir, f"scene_{scene_idx}_t{t}.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved {save_path}")
            img_count += 1

    if img_count == 0:
        print("没有生成任何图：可能该数据集 node_type 不叫 PEDESTRIAN，或这些 timestep 没有可预测节点。")
        print("你可以尝试：减小 --stride，或指定正确的 --node_type。")
    else:
        print(f"共生成 {img_count} 张图片，保存在目录：{args.output_dir}")


if __name__ == "__main__":
    main()