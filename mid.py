import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter

from tqdm.auto import tqdm
import pickle

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation


class MID():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self.global_step = 0
        self._build()

    def train(self):
        try:
            for epoch in range(1, self.config.epochs + 1):
                self.train_dataset.augment = self.config.augment
                for node_type, data_loader in self.train_data_loader.items():
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        self.optimizer.zero_grad()
                        train_loss = self.model.get_loss(batch, node_type)
                        pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                        train_loss.backward()
                        self.optimizer.step()

                        if getattr(self, "log_writer", None) is not None:
                            self.log_writer.add_scalar(
                                f"train/{str(node_type)}/loss",
                                float(train_loss.item()),
                                self.global_step
                            )
                            self.log_writer.add_scalar(
                                "train/lr",
                                float(self.optimizer.param_groups[0]["lr"]),
                                self.global_step
                            )
                            if self.global_step % 50 == 0:
                                self.log_writer.flush()

                        self.global_step += 1

                self.train_dataset.augment = False
                if epoch % self.config.eval_every == 0:
                    self.model.eval()

                    node_type = "PEDESTRIAN"
                    eval_ade_batch_errors = []
                    eval_fde_batch_errors = []

                    ph = self.hyperparams['prediction_horizon']
                    max_hl = self.hyperparams['maximum_history_length']

                    for i, scene in enumerate(self.eval_scenes):
                        print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                        for t in tqdm(range(0, scene.timesteps, 10)):
                            timesteps = np.arange(t, t + 10)
                            batch = get_timesteps_data(
                                env=self.eval_env,
                                scene=scene,
                                t=timesteps,
                                node_type=node_type,
                                state=self.hyperparams['state'],
                                pred_state=self.hyperparams['pred_state'],
                                edge_types=self.eval_env.get_edge_types(),
                                min_ht=7,
                                max_ht=self.hyperparams['maximum_history_length'],
                                min_ft=12,
                                max_ft=12,
                                hyperparams=self.hyperparams
                            )
                            if batch is None:
                                continue

                            test_batch = batch[0]
                            nodes = batch[1]
                            timesteps_o = batch[2]
                            traj_pred = self.model.generate(
                                test_batch,
                                node_type,
                                num_points=12,
                                sample=20,
                                bestof=True
                            )  # B * 20 * 12 * 2

                            predictions = traj_pred
                            predictions_dict = {}
                            for j, ts in enumerate(timesteps_o):
                                if ts not in predictions_dict:
                                    predictions_dict[ts] = dict()
                                predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))

                            batch_error_dict = evaluation.compute_batch_statistics(
                                predictions_dict,
                                scene.dt,
                                max_hl=max_hl,
                                ph=ph,
                                node_type_enum=self.eval_env.NodeType,
                                kde=False,
                                map=None,
                                best_of=True,
                                prune_ph_to_future=True
                            )

                            eval_ade_batch_errors = np.hstack(
                                (eval_ade_batch_errors, batch_error_dict[node_type]['ade'])
                            )
                            eval_fde_batch_errors = np.hstack(
                                (eval_fde_batch_errors, batch_error_dict[node_type]['fde'])
                            )

                    ade = np.mean(eval_ade_batch_errors)
                    fde = np.mean(eval_fde_batch_errors)

                    if self.config.dataset == "eth":
                        ade = ade / 0.6
                        fde = fde / 0.6
                    elif self.config.dataset == "sdd":
                        ade = ade * 50
                        fde = fde * 50

                    print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
                    self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

                    if getattr(self, "log_writer", None) is not None:
                        self.log_writer.add_scalar("eval/ade", float(ade), epoch)
                        self.log_writer.add_scalar("eval/fde", float(fde), epoch)
                        self.log_writer.flush()

                    # Saving model
                    checkpoint = {
                        'encoder': self.registrar.model_dict,
                        'ddpm': self.model.state_dict()
                    }
                    torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))

                    self.model.train()
        finally:
            if getattr(self, "log_writer", None) is not None:
                self.log_writer.flush()
                self.log_writer.close()

    def eval(self, sampling, step):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling} Stride: {step}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']

        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                batch = get_timesteps_data(
                    env=self.eval_env,
                    scene=scene,
                    t=timesteps,
                    node_type=node_type,
                    state=self.hyperparams['state'],
                    pred_state=self.hyperparams['pred_state'],
                    edge_types=self.eval_env.get_edge_types(),
                    min_ht=7,
                    max_ht=self.hyperparams['maximum_history_length'],
                    min_ft=12,
                    max_ft=12,
                    hyperparams=self.hyperparams
                )
                if batch is None:
                    continue
                test_batch = batch[0]
                nodes = batch[1]
                timesteps_o = batch[2]
                traj_pred = self.model.generate(
                    test_batch,
                    node_type,
                    num_points=12,
                    sample=20,
                    bestof=True,
                    sampling=sampling,
                    step=step
                )  # B * 20 * 12 * 2

                predictions = traj_pred
                predictions_dict = {}
                for j, ts in enumerate(timesteps_o):
                    if ts not in predictions_dict.keys():
                        predictions_dict[ts] = dict()
                    predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))

                batch_error_dict = evaluation.compute_batch_statistics(
                    predictions_dict,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    node_type_enum=self.eval_env.NodeType,
                    kde=False,
                    map=None,
                    best_of=True,
                    prune_ph_to_future=True
                )

                # --- Visualization Start ---
                try:
                    from evaluation.trajectory_utils import prediction_output_to_trajectories
                    from evaluation.visualization.visualization import plot_trajectories
                    import matplotlib.pyplot as plt
                    
                    p_dict, h_dict, f_dict = prediction_output_to_trajectories(
                        predictions_dict, scene.dt, max_hl, ph, prune_ph_to_future=True
                    )
                    figure_dir = f"results/figures/{self.config.dataset}"
                    os.makedirs(figure_dir, exist_ok=True)
                    
                    for ts in p_dict.keys():
                        if len(p_dict[ts]) > 0:
                            fig, ax = plt.subplots(figsize=(8, 8))
                            plot_trajectories(ax, p_dict[ts], h_dict[ts], f_dict[ts], line_alpha=0.2)
                            ax.set_aspect('equal')
                            ax.set_title(f"Scene {scene.name} - Timestep {ts}")
                            plt.savefig(f"{figure_dir}/scene_{scene.name}_t_{ts}.png", bbox_inches='tight', dpi=150)
                            plt.close(fig)
                except Exception as e:
                    print(f"Visualization failed: {e}")
                # --- Visualization End ---
                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))

        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)

        if self.config.dataset == "eth":
            ade = ade / 0.6
            fde = fde / 0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50
        print(f"Sampling: {sampling} Stride: {step}")
        print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
        # self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        # self._build_offline_scene_graph()
        # pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments", self.config.exp_name)
        os.makedirs(self.model_dir, exist_ok=True)

        self.tb_dir = osp.join(self.model_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)

        self.log_writer = SummaryWriter(log_dir=self.tb_dir)
        self.log_writer.add_text("meta/exp_name", str(self.config.exp_name), 0)
        self.log_writer.flush()

        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir, self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir, self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam(
            [
                {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                {'params': self.model.parameters()}
            ],
            lr=self.config.lr
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):
        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim // 2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim // 2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim // 2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim // 2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(
                osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"),
                map_location="cpu"
            )

            self.registrar.load_models(self.checkpoint['encoder'])

        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

    def _build_encoder(self):
        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")

        self.encoder.set_environment(self.train_env)
        # 关键：把同一个 writer 传下去，确保内部 summarize 写到事件文件
        if hasattr(self.encoder, "set_log_writer"):
            self.encoder.set_log_writer(self.log_writer)
        self.encoder.set_annealing_params()

    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder=self.encoder)

        self.model = model.cuda()
        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        self.train_dataset = EnvironmentDataset(
            train_env,
            self.hyperparams['state'],
            self.hyperparams['pred_state'],
            scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
            node_freq_mult=self.hyperparams['node_freq_mult_train'],
            hyperparams=self.hyperparams,
            min_history_timesteps=1,
            min_future_timesteps=self.hyperparams['prediction_horizon'],
            return_robot=not self.config.incl_robot_node
        )
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(
                node_type_data_set,
                collate_fn=collate,
                pin_memory=True,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.preprocess_workers
            )
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(
                self.eval_env,
                self.hyperparams['state'],
                self.hyperparams['pred_state'],
                scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                hyperparams=self.hyperparams,
                min_history_timesteps=self.hyperparams['minimum_history_length'],
                min_future_timesteps=self.hyperparams['prediction_horizon'],
                return_robot=not config.incl_robot_node
            )
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(
                    node_type_data_set,
                    collate_fn=collate,
                    pin_memory=True,
                    batch_size=config.eval_batch_size,
                    shuffle=True,
                    num_workers=config.preprocess_workers
                )
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(
                    self.train_env.attention_radius,
                    self.hyperparams['edge_addition_filter'],
                    self.hyperparams['edge_removal_filter']
                )
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(
                    self.eval_env.attention_radius,
                    self.hyperparams['edge_addition_filter'],
                    self.hyperparams['edge_removal_filter']
                )
                print(f"Created Scene Graph for Evaluation Scene {i}")