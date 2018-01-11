# Microgrid-Optimization
An optimization routine for minimizing electricity costs of a microgrid, consisting of a solar panel array and energy storage system.

A full report of this project is included in OptReport.pdf

* The python script can be easily modified to work for other instances. 
* To change the expected load profile of the microgrid, the ```load``` array should be modified accordingly.
* To change the electricity spot prices corresponding to each hour, the ```price``` array should be modified accordingly.
* To change the expected solar power hourly capacity, the ```solar``` array should be modified accordingly.
* To change the capacity of the energy storage system, ```battery_cap``` should be changed accordingly.
* To change the power limit of the energy storage system, ```bat_pwr_rating``` should be modified accourdingly.

Each of these variables are defined and clearly labeld near the top of the script, and their units are given.
