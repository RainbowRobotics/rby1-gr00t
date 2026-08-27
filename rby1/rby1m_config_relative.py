from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


_ACTION_HORIZON = 40  # matches model_config.action_horizon (see gr00t/configs/model/gr00t_n1d7.py)

rby1m_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_front_head",
            "cam_right_wrist",
            "cam_left_wrist",
        ],  # Must match a key in meta/modality.json under "video"
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_arm",
            "left_arm",
            "right_gripper",
            "left_gripper",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(
            range(0, _ACTION_HORIZON)
        ),  # Action prediction horizon (Action Chunk Size)
        modality_keys=[
            "right_arm",  # Must match keys in meta/modality.json under "action"
            "left_arm",
            "right_gripper",
            "left_gripper",
        ],
        action_configs=[
            # Assign each ActionConfig in the same order as modality_keys above
            # right_arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Left arm
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Using relative actions will lead to smoother actions, but might suffer from drifting.
            # If you want to use relative actions, please make sure the state and action stored in the dataset are absolute, and the absolute to relative will be handled in the processor.
            # Action Type (Cartesian vs. joint space):
            # - `EEF`: End-effector/Cartesian space control (Expecting a 9-dimensional vector: x, y, z positions + rotation 6D)
            # - `NON_EEF`: Joint space control and other non-EEF control spaces (joint angles, positions, gripper positions, etc.)
            # right_gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # absolute control of the gripper
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,  # absolute control of the gripper
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "annotation.human.task_description"
        ],  # Must match annotation keys in meta/modality.json
    ),
}

register_modality_config(rby1m_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
