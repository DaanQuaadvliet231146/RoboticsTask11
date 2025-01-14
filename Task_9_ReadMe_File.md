# Task 9: Simulation Environment for Opentrons OT-2

## Environment Setup

### Dependencies
To set up the simulation environment, the following dependencies are installed:

1. **Python** (Version 3.8 or higher)
2. **PyBullet**
   ```bash
   pip install pybullet
   ```
3. **The OT2 Twin repositary.**
   ```
   git clone https://github.com/BredaUniversityADSAI/Y2B-2023-OT2_Twin.git
   ```
### Usage
To run the simulation, I used the provided Python script `task_9.ipynb`. This script:
- Moves the OT-2 pipette to the corners of its working envelope.
- Records and prints the coordinates of each corner.
---

## Working Envelope
The working envelope defines the cube-shaped region where the pipette can operate. The recorded coordinates for the pipette's eight corners are:

1. Corner 1: [1.0, 1.0, 1.0]
2. Corner 2: [-1.0, 1.0, 1.0]
3. Corner 3: [-1.0, -1.0, 1.0]
4. Corner 4: [1.0, -1.0, 1.0]
5. Corner 5: [1.0, 1.0, -1.0]
6. Corner 6: [-1.0, 1.0, -0.1]
7. Corner 7: [-1.0, -1.0, -1.0]
8. Corner 8: [1.0, -1.0, -1.0]

These coordinates were determined by moving the pipette using predefined velocities and recording its position at each step.

---

## Code Overview

### `task_9.ipynb`
This Python script:
- Initializes the OT-2 Digital Twin simulation environment.
- Moves the pipette to predefined positions.
- Logs the pipette’s coordinates for each corner of the working envelope.

**Key Functions:**

1. `move_and_record(sim)`: Moves the pipette to each corner of the cube and logs the positions.
2. Simulation initialization and reset.

## Gif
I made a gif of the simulation: `GIF_DATALAB_TASK9_UWU`. 

## Results
Results Were as follows:
1. Corner 1: X=0.2531, Y=0.2195, Z=0.2895
2. Corner 2: X=-0.1469, Y=0.2195, Z=0.2895
3. Corner 3: X=-0.1872, Y=-0.1708, Z=0.2895
4. Corner 4: X=0.2130, Y=-0.1705, Z=0.2896
5. Corner 5: X=0.2531, Y=0.2198, Z=0.1691
6. Corner 6: X=-0.1468, Y=0.2201, Z=0.1692
7. Corner 7: X=-0.1871, Y=-0.1711, Z=0.1691
8. Corner 8: X=0.2128, Y=-0.1705, Z=0.1692

### Dimensions of the Working Envelope
- **Width**: 0.4403 units (X-axis)
- **Depth**: 0.3912 units (Y-axis)
- **Height**: 0.1205 units (Z-axis)
