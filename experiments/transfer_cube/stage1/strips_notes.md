# Frame strips, what I saw

All 40 strips (30 failures, 10 successes) were looked at on 2026-11-24, enlarged from the saved 200 by 150 pixel frames. Each has 10 frames, 44 steps apart (steps 0 to 396). These are visual impressions at low resolution. A lift shorter than 44 steps can fall between frames. No number in diagnosis.md rests on them.

| Seed | Label | Read |
|---|---|---|
| 20455, 20628 | class 0 | Closes beside the cube, pushes it a little, then runs the handover motion with an empty gripper |
| 20671, 20832 | class 0 | Goes back to the cube at least once, cube stays on the table, handover motion with an empty gripper in between or at the end |
| 20003, 20108, 20587, 20951, 20967 | class 1 | Handover motion with no visible cube in the gripper, then the right arm returns to the cube |
| 20563 | class 1 | Handover motion to the end with an empty gripper, cube not visible in the last frames |
| 20407, 20549, 20631, 20760 | class 1 | Repeated approaches, no visible lift |
| 20129, 20180, 20590 | class 2, not lifted | Handover motion with an empty gripper, cube back on or still on the table |
| 20084, 20417, 20710 | class 2, not lifted | Repeated approaches, no visible lift |
| 20586 | class 2, not lifted | Right gripper stays beside the cube all episode |
| 20043, 20134, 20318 | class 2, lifted | Cube carried off to the side of the fingertips, the left gripper closes next to it, the right gripper still has the cube at the end |
| 20022, 20414, 20974 | class 2, lifted | Several failed approaches, cube gripped only in the last frame. A late lift |
| 20317 | class 2, lifted | Cube seen in the air beside the gripper in one frame, then on the table |
| 20674, 20961 | class 2, lifted | No lift visible in the 10 frames. Looks like a grasp failure, the label says lifted |
| 20133, 20390, 20404, 20656, 20684, 20690, 20708, 20837, 20874, 20938 | success, class 4b | One approach, lift, handover, cube in the left gripper and the right gripper away at the end. 20837 is later than the rest |

All 10 successes are class 4b, because that is how they were picked, so they cannot show whether the strict rule misses good class 4a episodes. I did not watch any class 4a episode.
