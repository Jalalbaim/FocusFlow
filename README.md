# FocusFlow

Standard FlowEdit operates globally. It calculates a single "velocity field" (direction of change) for the entire image.

Our Idea is to modify the ODE solver to compute a weighted sum of velocities. This ensures the background follows the source trajectory while the foreground follows the target trajectory.

$$V_{t}^{final} = \underbrace{\mathbf{M} \odot V_{t}^{target}}_{\text{Apply edit here}} + \underbrace{(1 - \mathbf{M}) \odot V_{t}^{source}}_{\text{Force original structure here}}$$

Where:

- $\mathbf{M}$ is a spatial mask ($0$ to $1$) defining the edit region.

- $V_{t}^{target}$ is the velocity predicted using the Target Prompt (e.g., "A tiger sitting").

- $V_{t}^{source}$ is the velocity predicted using the Source Prompt (e.g., "A cat sitting").

- $\odot$ is element-wise multiplication.

Papers:

    FlowEdit: https://arxiv.org/pdf/2412.08629

    DiffEdit: https://arxiv.org/pdf/2210.11427

    RegionDrag: https://arxiv.org/pdf/2407.18247

    FlowAlign: https://arxiv.org/pdf/2505.23145
