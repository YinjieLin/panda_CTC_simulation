This project is an example of Franka Panda simulation based on Mujoco.
## Dependencies
- Mujoco200
- Eigen 3

## Usage
Modify the ${MJPRO_PATH} in CMakeLists.txt file of yours. Add the license mjkey.txt of yours to ./build.
Then,
```
cd build
cmake ..
make
```

Run the simulation with
```
./panda_sim_mujoco ../models/my_panda.xml
```