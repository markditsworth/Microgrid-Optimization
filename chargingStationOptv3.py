#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:50:25 2017

@author: markditsworth
"""

import scipy.optimize as opt
import numpy as np
import pandas as pd
df = pd.DataFrame()
# Load Profile (hourly)
load = np.array([0,13,15,20,22,13,8,9,0,0,0,0])         #load in kW
load = load * 1000                                      #load to W
df['Load'] = load
# Electricty Prices (hourly)
price = np.array([10,15,23,30,33,29,13,10,9,8,7,9])   #cents/kWh
price = price / 100.0                                   #$/kWh
price = price / 1000.0                                  #$/Wh
df['Price'] = price
# Solar Availability (hourly)
solar = np.array([0,0,9,10,19,22,14,0,0,0,0,0])         #solar availability (kW)
solar = solar * 1000

cost = np.dot(load,price.T)
print('Base Cost: $%.2f'%cost)

df['Solar Available'] = solar
# Demand = Power not provided by solar
demand = np.subtract(load,solar)                        #demand in W
df['Demand']=demand
# Demand clipped at 0 W to prevent negative demand
demand = demand.clip(min=0)

cost = np.dot(demand,price.T)
print('Cost with Solar: $%.2f'%cost)

# Solar power actually used (accounting for periods when solar > load)
solar_use_station = load - demand
df['Solar Use Station'] = solar_use_station
df['Solar Use Battery'] = np.zeros(len(load))
# Battery Capacity in Wh
battery_cap = 8 * 1000

# Battery power rating in W
bat_pwr_rating = 5*1000

# Initialized guesses for battery power use (hourly)
x = np.array([500]*12)

def SOC(battery_use_array):
    soc_array = np.array([battery_cap]*len(battery_use_array))
    use = np.cumsum(battery_use_array)
    use = np.roll(use,1)
    use[0]=0
    soc_array = np.subtract(soc_array,use)
    return soc_array

def GRID(battery_use_array):
    grid_use = np.subtract(demand,battery_use_array)
    return grid_use

def COST(battery_use_array):
    gridUse = GRID(battery_use_array)
    elecCost = np.dot(gridUse,price.T)
    return elecCost

# [1 1 ... 1] to be dotted with x to get throughput
A = np.ones(12)
# Upper bound for capacity
b = battery_cap

# Init. bounds

#bnds = [[0,1],[0,1]] * 6
bnds = []
upperBnds = demand.clip(max=bat_pwr_rating)
for x in upperBnds:
    bnds.append([0,x])

soln = opt.linprog(-1*price,A_ub=np.ones(len(price)),b_ub=battery_cap,bounds=bnds)

#print(soln)
bat_use = soln.x
df['Battery Use'] = bat_use
df['Battery Charge'] = np.zeros(len(price))
bat_soc = SOC(bat_use)
df['Battery SOC'] = bat_soc
grid = load - solar_use_station - bat_use
df['Grid']=grid
cost = np.dot(grid,price.T)
print("Optimized Cost: $%.2f"%cost)

def replace(fromDF,toDF):
    i = fromDF.index.values
    toDF.loc[i,:] = fromDF.loc[:,:]
    return toDF

#################################################################
# Allow for charging of the battery with excess solar
#################################################################
# Construct DataFrame of times when battery is neither being used, nor fully charged
df_sub = df[(df['Battery Use']==0) & (df['Battery SOC']<battery_cap)]
# get slice of df_sub where there is a negative demand of power
df_temp = df_sub[df_sub['Demand']<0]
end_index = df_temp.index.values[-1] +1
# Get array of excess solar power
excess_solar = df_temp['Demand'].values
# Limit this power by the battery's rating
excess_solar = excess_solar.clip(min=-1*bat_pwr_rating)
# Excess Solar power into battery
df_temp.loc[:,'Battery Charge'] = excess_solar
# Record solar power used to charge battery
df_temp.loc[:,'Solar Use Battery'] = -1*excess_solar
# place df_temp back within df_sub
df_sub = replace(df_temp,df_sub)
# place df_sub back within df
df = replace(df_sub,df)

total_bat_use = np.add(df['Battery Use'].values,df['Battery Charge'].values)
# Recalaculate SOC
df['Battery SOC'] = SOC(total_bat_use)
# find where SOC > Capacity
df_temp = df[df['Battery SOC'] > battery_cap]
# get indexes of over charging
SOCindex = df_temp.index.values
Chargeindex = SOCindex -1
# get ammount overcharged
overcharge = df.loc[SOCindex[0],'Battery SOC']
# fix initial overcharge
df.loc[Chargeindex[0],'Battery Charge'] = -1*(overcharge - battery_cap)
df.loc[Chargeindex[0],'Solar Use Battery']=df.loc[Chargeindex[0],'Solar Use Battery']-(overcharge - battery_cap)
# remove additional overcharges
df.loc[Chargeindex[1:],'Battery Charge'] = 0
# recalculate SOC
total_bat_use = np.add(df['Battery Use'].values,df['Battery Charge'].values)
df['Battery SOC'] = SOC(total_bat_use)

#################################################################
# Re-optimize after excess solar is used to charge the battery
#################################################################
#print(end_index)
new_bat_cap = df.loc[end_index,'Battery SOC']
new_price = price[end_index:]
new_demand = demand[end_index:]
bnds = []
upperBnds = new_demand.clip(max=bat_pwr_rating)
for x in upperBnds:
    bnds.append([0,x])

soln = opt.linprog(-1*new_price,A_ub=np.ones(len(new_price)),b_ub=new_bat_cap,bounds=bnds)

new_bat_use = soln.x
df.loc[end_index:,'Battery Use'] = new_bat_use

total_bat_use = np.add(df['Battery Use'].values,df['Battery Charge'].values)
df['Battery SOC'] = SOC(total_bat_use)
df['Grid'] = load - solar_use_station - df['Battery Use'].values

new_cost = np.dot(df['Grid'].values,price.T)
print('Re-optimized Cost: $%.2f'%new_cost)

#################################################################
# Charge Battery
#################################################################
index = np.nonzero(total_bat_use)
index = int(index[0][-1] + 1)
new_price = price[index:]
newSOC = df.loc[index,'Battery SOC']
soln = opt.linprog(new_price,A_eq=np.array([np.ones(len(new_price))]),b_eq=battery_cap-newSOC,bounds=[0,bat_pwr_rating])
grid_to_bat = np.zeros(len(price))
grid_to_bat[index:] = soln.x

df.loc[:,'Battery Charge'] = np.add(df['Battery Charge'].values,-1*grid_to_bat)
total_bat_use = np.add(df['Battery Use'].values,df['Battery Charge'].values)
df['Battery SOC'] = SOC(total_bat_use)

added_cost = np.dot(soln.x,new_price)
print('Cost With Recharge: $%.2f'%(new_cost + added_cost))

# Visualize Results
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
x = np.arange(0,12,1)

plt.figure(1)
plt.subplot(411)
plt.plot(x,load,color='black',label='Load')
plt.ylabel('Power (W)')
plt.text(0,20000,'Load')

#plt.subplot(312)
plt.stackplot(x,[df['Solar Use Station'].values,
                 df['Battery Use'].values,df['Grid'].values],colors=['r','g','c'])

#plt.ylabel('Power (W)')
#plt.text(0,20000,'Sources')
red = mpatches.Patch(color='red',label='Solar')
green = mpatches.Patch(color='green',label='Battery')
cyan = mpatches.Patch(color='c',label='Grid')
plt.legend(handles=[red,green,cyan],loc='upper right')

plt.subplot(412)
plt.stackplot(x,[df['Solar Use Battery'].values,grid_to_bat],colors=['r','c'])
plt.ylabel('Power (W)')
plt.text(0,4000,'Power to Bat (W)')
red = mpatches.Patch(color='red',label='Solar')
cyan = mpatches.Patch(color='c', label='Grid')
plt.legend(handles=[red,cyan],loc='upper right')

plt.subplot(413)
plt.plot(x,total_bat_use,color='black')
plt.ylabel('Power (W)')
plt.text(0,4000,'Battery Use')

plt.subplot(414)
plt.plot(x,df['Battery SOC'].values)
plt.ylabel('SOC (Wh)')
plt.text(0,6000,'Battery SOC')
plt.legend()
plt.savefig('Optv2.png',dpi=300)
plt.show()
