"""RANS task implementations."""

from .go_to_position import GoToPositionTask
from .go_to_pose import GoToPoseTask
from .track_linear_velocity import TrackLinearVelocityTask
from .track_linear_angular_velocity import TrackLinearAngularVelocityTask

TASK_REGISTRY = {
    "GoToPosition": GoToPositionTask,
    "GoToPose": GoToPoseTask,
    "TrackLinearVelocity": TrackLinearVelocityTask,
    "TrackLinearAngularVelocity": TrackLinearAngularVelocityTask,
}

__all__ = [
    "GoToPositionTask",
    "GoToPoseTask",
    "TrackLinearVelocityTask",
    "TrackLinearAngularVelocityTask",
    "TASK_REGISTRY",
]
