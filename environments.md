# Environments

Enviornments are based on robosuite, and for more information on the
details of robosuite, including how to create an environment, please
refer to [robosuite doc](https://robosuite.ai/docs)

We describe the tasks included in this repository.

## Task Descriptions

We provide a brief description of each environment included in this repo. The current focuse is on manipulation environments.

### ToolUse
![env_tool_use](./images/env_tool_use.png)
- **Description**: A simple tabletop domain to place a hammer into the cabinet
  and close the cabinet.
- **Robot-type**: Single-Arm
- **Task-type**: Single-Task
- **Has rewards**: Yes
- **Has demonstrations**: Yes
  - **No. of Demonstrations**: 100
  - **Robot**: Panda
- **Action Space**: OSC_POSITION (3) + Gripper (1)
- **Observations**: Workspace images, Eye-in-Hand images, Proprioception
- **Reference**: [Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation](paper_link)

### HammerPlace
![env_hammer_place](./images/env_hammer_place.png)
- **Description**: A simple tabletop domain to place a hammer into the cabinet
  and close the cabinet.
- **Robot-type**: Single-Arm
- **Task-type**: Single-Task
- **Has rewards**: Yes
- **Has demonstrations**: Yes
  - **No. of Demonstrations**: 100
  - **Robot**: Panda
- **Action Space**: OSC_POSITION (3) + Gripper (1)
- **Observations**: Workspace images, Eye-in-Hand images, Proprioception
- **Reference**: [Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation](paper_link)

### Kitchen
![env_kitchen](./images/env_kitchen.png)
- **Description**: A simple tabletop domain to place a hammer into the cabinet
  and close the cabinet.
- **Robot-type**: Single-Arm
- **Task-type**: Single-Task
- **Has rewards**: Yes
- **Has demonstrations**: Yes
  - **No. of Demonstrations**: 100
  - **Robot**: Panda
- **Action Space**: OSC_POSITION (3) + Gripper (1)
- **Observations**: Workspace images, Eye-in-Hand images, Proprioception
- **Reference**: [Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation](paper_link)


### MultitaskKitchenDomain
![env_multitask_kitchen](./images/env_multi_kitchen.png)
- **Description**: A simple tabletop domain to place a hammer into the cabinet
  and close the cabinet.
- **Robot-type**: Single-Arm
- **Task-type**: Multi-Task
  - **No. of Tasks**: 3
- **Has rewards**: Yes
- **Has demonstrations**: Yes
  - **No. of Demonstrations in total**: 360
  - **No. of Demonstrations each task**: 120
  - **Robot**: Panda
- **Action Space**: OSC_POSITION (3) + Gripper (1)
- **Observations**: Workspace images, Eye-in-Hand images, Proprioception
- **Reference**: [Bottom-Up Skill Discovery from Unsegmented Demonstrations for Long-Horizon Robot Manipulation](paper_link)


