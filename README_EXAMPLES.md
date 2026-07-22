# RL Reach, Pick and Place taught examples
----------------
This document presents all the results of models trained with these scripts 

----------------


## Reach

<table>
<tr>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\REACH_PPO_GIF.gif" width="250"><br>
PPO
</td>
<td align="center">
<img src="Gymnasium-Robotics-main/gifs_images/REACH_SAC_GIF.gif" width="250"><br>
SAC
</td>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\REACH_TQC_GIF.gif" width="250"><br>
TQC
</td>
</tr>
</table>
<table>
<tr>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\REACH_PPO.png" width="250"><br>
PPO
</td>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\REACH_SAC.png" width="250"><br>
SAC
</td>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\REACH_TQC.png" width="250"><br>
TQC
</td>
</tr>
</table>

## Pick and Place

<table>
<tr>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\PGP_PPO_GIF.gif" width="250"><br>
PPO
</td>
<td align="center">
<img src="Gymnasium-Robotics-main/gifs_images/PGP_SAC_GIF.gif" width="250"><br>
SAC
</td>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\PGP_TQC_GIF.gif" width="250"><br>
TQC
</td>
</tr>
</table>
<table>
<tr>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\PGP_PPO.png" width="250"><br>
PPO
</td>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\PGP_SAC.png" width="250"><br>
SAC
</td>
<td align="center">
<img src="Gymnasium-Robotics-main\gifs_images\PGP_TQC.png" width="250"><br>
TQC
</td>
</tr>
</table>

### Notes

The biggest issue with getting P&P to work is verifying that the object is grasped as to start the lift. The agent has started learning that it's better to just hover near the block than to attempt to pick it up.