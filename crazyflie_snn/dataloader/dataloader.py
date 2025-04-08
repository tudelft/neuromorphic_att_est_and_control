import os

import torch
import pandas as pd
import numpy as np

class DatasetControl(torch.utils.data.Dataset):
    def __init__(self, training_folder, options):
        self.options = options
        self.source = options["source"]
        self.remove_integral = options["remove_integral"]
        if "remove_initial_value" in options.keys():
            self.remove_initial_value = options["remove_initial_value"]
        else: self.remove_initial_value = False
        # Check if length of input scaling is equal to number of input columns
        if len(options["input_scaling"]) != len(options["input_columns"]):
            raise Exception("Length of input scaling does not match number of input columns")
        self.input_scaling = options["input_scaling"]
        # Check if length of target scaling is equal to number of target columns
        if len(options["target_scaling"]) != len(options["target_columns"]):
            raise Exception("Length of target scaling does not match number of target columns")
        self.target_scaling = options["target_scaling"]
        if "input_columns" in options:
            self.input_columns = options["input_columns"]
            self.target_columns = options["target_columns"]
        else:
            self.input_columns = None
            self.target_columns = None

        # Read all datasets in the given training folder
        self.dataset_list = []
        for (_, _, filenames) in os.walk(training_folder, topdown=True):
            for filename in filenames:
                if self.source == "pybullet":
                    dataset = np.load(os.path.join(training_folder, filename))
                elif self.source == "crazyflie":
                    dataset = pd.read_csv(os.path.join(training_folder, filename))
                elif self.source == "px4":
                    dataset = pd.read_csv(os.path.join(training_folder, filename))
                self.dataset_list.append(dataset)

        self.indices = range(len(self.dataset_list))
        # Create a list of custom (tuple) indices
        self.indices = []
        shift = self.options["target_shift"] if "target_shift" in self.options else 0
        for i_dataset, dataset in enumerate(self.dataset_list):
            if self.source == "pybullet":
                dataset_length = dataset["states"].shape[2]
            elif self.source == "crazyflie":
                dataset_length = len(dataset)
            elif self.source == "px4":
                dataset_length = len(dataset)
            for idx in range(
                25,
                dataset_length - options["seq_length"] - shift,
                options["inter_seq_dist"],
            ):
                self.indices.append((i_dataset, idx))

        print(
            f"loaded {len(self.dataset_list)} datasets with a total of {len(self.indices)} sequences"
        )

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.indices)

    def __getitem__(self, idx):
        """Generates a sequence of data"""
        i_dataset, i_start = self.indices[idx]
        if self.source == "pybullet":
            input_values, target_values = self.get_item_pybullet(i_dataset, i_start)
        elif self.source == "crazyflie":
            input_values, target_values = self.get_item_crazyflie(i_dataset, i_start)
        elif self.source == "px4":
            input_values, target_values = self.get_item_px4(i_dataset, i_start)

        return input_values.to(dtype=torch.float), target_values.to(dtype=torch.float)

    def get_item_pybullet(self, i_dataset, i_start):
        # Could still be implemented for training from sim
        return 0

    def get_item_px4(self, i_dataset, i_start):
        # Could still be implemented for flying bigger drone
        pass
    
    def get_item_crazyflie(self, i_dataset, i_start):
        seq_lgth = self.options["seq_length"]
        # Set inputs and outputs from columns
        if self.input_columns is None:
            input_columns = ['gyro.x', 'gyro.y', 'gyro.z', 'acc.x', 'acc.y', 'acc.z', 'controller.roll', 'controller.pitch']
            # Check if output has pid_rate.roll_output column to match newer datasets
            if 'pid_rate.roll_output' in self.dataset_list[i_dataset].columns:
                output_columns = ['pid_rate.roll_output', 'pid_rate.pitch_output']
            else:
                output_columns = ['controller.cmd_roll', 'controller.cmd_pitch']
        else:
            input_columns = self.input_columns.copy()
            if 'pid_attitude.roll_output' in self.input_columns:
                if 'pid_attitude.roll_output' not in self.dataset_list[i_dataset].columns:
                    input_columns[input_columns.index('pid_attitude.roll_output')] = 'controller.rollRate'
                if 'pid_attitude.pitch_output' not in self.dataset_list[i_dataset].columns:
                    input_columns[input_columns.index('pid_attitude.pitch_output')] = 'controller.pitchRate'
            output_columns = self.target_columns.copy()
            if 'pid_rate.roll_output' in self.target_columns:
                if 'pid_rate.roll_output' not in self.dataset_list[i_dataset].columns:
                    output_columns[output_columns.index('pid_rate.roll_output')] = 'controller.cmd_roll'
                if 'pid_rate.pitch_output' not in self.dataset_list[i_dataset].columns:
                    output_columns[output_columns.index('pid_rate.pitch_output')] = 'controller.cmd_pitch'
            if 'pid_attitude.roll_output' in self.target_columns:
                if 'pid_attitude.roll_output' not in self.dataset_list[i_dataset].columns:
                    output_columns[output_columns.index('pid_attitude.roll_output')] = 'controller.rollRate'
                if 'pid_attitude.pitch_output' not in self.dataset_list[i_dataset].columns:
                    output_columns[output_columns.index('pid_attitude.pitch_output')] = 'controller.pitchRate'


        inputs = torch.from_numpy(self.dataset_list[i_dataset].loc[i_start:i_start+seq_lgth - 1, input_columns].to_numpy())
        inputs = inputs.to(dtype=torch.float)
        # Scale inputs
        for i in range(len(input_columns)):
            inputs[:, i] = inputs[:, i] * self.input_scaling[i]

        # Shift target if necessary
        shift = self.options["target_shift"] if "target_shift" in self.options else 0
        target = torch.from_numpy(self.dataset_list[i_dataset].loc[i_start+shift:i_start+shift+seq_lgth - 1, output_columns].to_numpy())

        # Remove integral if necessary
        if self.remove_integral:
            columns_to_adjust = ["pid_attitude.roll_output", 
                                 "pid_attitude.pitch_output", 
                                 "pid_rate.roll_output", 
                                 "pid_rate.pitch_output", 
                                 "pid_rate.yaw_output", 
                                 "controller.rollRate", 
                                 "controller.pitchRate"]
            integral_columns = ["pid_attitude.roll_outI", 
                                "pid_attitude.pitch_outI", 
                                "pid_rate.roll_outI", 
                                "pid_rate.pitch_outI", 
                                "pid_rate.yaw_outI", 
                                "pid_attitude.roll_outI", 
                                "pid_attitude.pitch_outI"]
            for column, int_column in zip(columns_to_adjust, integral_columns):
                if column in output_columns:
                    target[:, output_columns.index(column)] = target[:, output_columns.index(column)] - torch.from_numpy(self.dataset_list[i_dataset].loc[i_start:i_start+seq_lgth - 1, int_column].to_numpy())
        elif self.remove_initial_value:
            # Remove only the first value of the integral
            columns_to_adjust = ["pid_attitude.roll_output", 
                                 "pid_attitude.pitch_output", 
                                 "pid_rate.roll_output", 
                                 "pid_rate.pitch_output", 
                                 "pid_rate.yaw_output", 
                                 "controller.rollRate", 
                                 "controller.pitchRate",
                                 "pid_rate.roll_outI",
                                 "pid_rate.pitch_outI",
                                 "pid_rate.yaw_outI"]
            integral_columns = ["pid_attitude.roll_outI", 
                                "pid_attitude.pitch_outI", 
                                "pid_rate.roll_outI", 
                                "pid_rate.pitch_outI", 
                                "pid_rate.yaw_outI", 
                                "pid_attitude.roll_outI", 
                                "pid_attitude.pitch_outI",
                                "pid_rate.roll_outI",
                                "pid_rate.pitch_outI",
                                "pid_rate.yaw_outI"]
            for column, int_column in zip(columns_to_adjust, integral_columns):
                if column in output_columns:
                    target[:, output_columns.index(column)] = target[:, output_columns.index(column)] - self.dataset_list[i_dataset].loc[i_start, int_column]
        target = target.to(dtype=torch.float)

        # Scale target
        for i in range(len(output_columns)):
            # If target column is an integral, remove initial value before scaling
            if self.remove_initial_value:
                if output_columns[i] in ["pid_rate.roll_outI", "pid_rate.yaw_outI", "pid_rate.yaw_outI", "pid_attitude.roll_outI", "pid_attitude.pitch_outI", "pid_attitude.yaw_outI"]:
                    target[:, i] = target[:, i] - target[0, i]
            target[:, i] = target[:, i] * self.target_scaling[i]

        if "target_integral" in self.options.keys():
            if self.options["target_integral"]:
                target = torch.cumsum(target, dim=0) * 0.002

        # Repeat data if necessary
        n_repeats = self.options["n_repeats"] if "n_repeats" in self.options else 1
        if n_repeats:
            inputs = inputs.repeat_interleave(n_repeats, dim=0)
            target = target.repeat_interleave(n_repeats, dim=0)

        # inputs = torch.zeros_like(inputs)
        # inputs[:, 1:] = 0
        return inputs, target.squeeze(0)
        


def load_datasets(config, generator=None):

    # Check if "Train" and "Validation" folders exist
    if not os.path.exists(os.path.join(config["data"]["dir"], "Train")):
        raise Exception("No training folder found")
    if not os.path.exists(os.path.join(config["data"]["dir"], "Validation")):
        raise Exception("No validation folder found")
    
    train_folder = os.path.join(config["data"]["dir"], "Train")
    validation_folder = os.path.join(config["data"]["dir"], "Validation")
    test_folder = os.path.join(config["data"]["dir"], "Test")

    if config["data"]["type"] == "control":
        train_dataset = DatasetControl(train_folder, config["data"])
        val_dataset = DatasetControl(validation_folder, config["data"])
        test_dataset = DatasetControl(test_folder, config["data"])

    train_smplr = torch.utils.data.RandomSampler(np.arange(len(train_dataset)), generator=generator)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                            shuffle=False,
                            batch_size=config["data"]["train_batch_size"],
                            sampler=train_smplr,
                            drop_last=True)

    val_smplr = torch.utils.data.RandomSampler(np.arange(len(val_dataset)), generator=generator)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=config["data"]["val_batch_size"],
                            sampler=val_smplr,
                            drop_last=True)
    
    test_smplr = torch.utils.data.RandomSampler(np.arange(len(test_dataset)), generator=generator)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                            shuffle=False,
                            batch_size=2,
                            sampler=test_smplr,
                            drop_last=True)
    
    
    return train_loader, val_loader, test_loader